// RFC 3711 - Secure Real-time Transport Protocol (SRTP) and
// RFC 3711 - Secure RTCP (SRTCP)
//
// This module implements:
//   * AES-128-CM encryption/decryption (RFC 3711 §4.1)
//   * HMAC-SHA1 authentication tags (RFC 3711 §4.2)
//   * AES-CM pseudo-random function for session-key derivation (RFC 3711 §4.3)
//   * Packet index estimation / rollover counter tracking (RFC 3711 §3.3.1)
//   * Replay-window protection for incoming packets (RFC 3711 §3.3)
//   * SRTP (RTP payload) and SRTCP (RTCP payload) protocol variants
//   * Packet reader/writer wrappers that plug into the existing rfc3550 I/O traits
//
// FIXME: add replay-protection for *outgoing* packets (two-time-pad guard)

use fixedbitset::FixedBitSet;
use handy_async::sync_io::{ReadExt, WriteExt};
use std::borrow::Cow;
use std::collections::BTreeMap;
use std::io::{Read, Write};
use trackable::*;
// NOTE(future): `trackable` is a legacy error-handling crate. Consider migrating to
// `thiserror` + `anyhow` across rtp once the current test/fix period is stable.
// See: https://crates.io/crates/thiserror  - rtp-internal change, no surface impact.

use crate::io::{ReadFrom, WriteTo};
use crate::rfc3550;
use crate::traits::{ReadPacket, RtcpPacket, RtpPacket, WritePacket};
use crate::types::{Ssrc, U48};
use crate::{ErrorKind, Result};

use aes::cipher::{BlockEncrypt, KeyInit, KeyIvInit, StreamCipher};
use aes::Aes128;
use ctr::Ctr128BE;
use hmac::{Hmac, Mac};

// Type aliases

/// HMAC-SHA1 type alias, kept private; used only through [`hmac_hash_sha1`].
type HmacSha1 = Hmac<sha1::Sha1>;

// Algorithm selectors

/// Encryption algorithm to use for this SRTP/SRTCP context.
/// Only AES-CM is implemented; AES-F8 and Null are present for future use
/// and RFC-completeness.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum EncryptionAlgorithm {
    #[default]
    AesCm,
    AesF8,
    Null,
}

/// Authentication algorithm. Only HMAC-SHA1 is implemented (RFC 3711 §4.2).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum AuthenticationAlgorithm {
    #[default]
    HmacSha1,
}

// Protocol trait

/// Abstracts over the SRTP (RTP) and SRTCP (RTCP) protocol differences.
///
/// Both protocols share the same AES-CM / HMAC-SHA1 machinery but differ in:
///   * which header bytes carry the SSRC and packet index
///   * how the authenticated byte span is constructed
///   * how the packet index is maintained / predicted
///   * which key-derivation labels are used (§4.3)
pub trait Protocol: Sized + Default {
    /// Packet index type: [`U48`] for SRTP, `u32` (31-bit) for SRTCP.
    type PacketIndex: Sized + Ord + Into<u64> + Copy;

    /// Key-derivation label for the session encryption key (§4.3).
    const ENC_KEY_LABEL: u8;
    /// Key-derivation label for the session authentication key.
    const AUTH_KEY_LABEL: u8;
    /// Key-derivation label for the session salt.
    const SALT_KEY_LABEL: u8;

    /// Extract the SSRC from a raw packet buffer.
    fn read_ssrc(packet: &[u8]) -> Result<Ssrc>;

    /// Estimate the full packet index for an *incoming* packet.
    fn determine_incoming_packet_index(
        context: &Context<Self>,
        ssrc_context: &SsrcContext<Self>,
        packet: &[u8],
    ) -> Result<Self::PacketIndex>;

    /// Determine the packet index for an *outgoing* packet.
    fn determine_outgoing_packet_index(
        context: &SsrcContext<Self>,
        packet: &[u8],
    ) -> Result<Self::PacketIndex>;

    /// Return the bytes that HMAC-SHA1 is computed over.
    /// For SRTP this appends the ROC; for SRTCP the full index is already in
    /// the packet, so the slice is returned unchanged.
    fn get_authenticated_bytes<'a>(
        context: &Context<Self>,
        index: Self::PacketIndex,
        auth_portion: &'a [u8],
    ) -> Result<Cow<'a, [u8]>>;

    /// Decrypt the ciphertext payload, returning a plaintext packet.
    fn decrypt(
        context: &Context<Self>,
        ssrc_context: &SsrcContext<Self>,
        packet: &[u8],
        index: Self::PacketIndex,
    ) -> Result<Vec<u8>>;

    /// Encrypt the plaintext payload, returning a ciphertext packet
    /// (without the auth tag - that is appended by `process_outgoing`).
    fn encrypt(
        context: &Context<Self>,
        ssrc_context: &SsrcContext<Self>,
        packet: &[u8],
        index: Self::PacketIndex,
    ) -> Result<Vec<u8>>;

    /// Update the highest received index / ROC after a packet is accepted.
    fn update_highest_recv_index(context: &mut SsrcContext<Self>, index: Self::PacketIndex);

    /// Update the highest sent index after a packet is emitted.
    fn update_highest_sent_index(context: &mut SsrcContext<Self>, index: Self::PacketIndex);
}

// Context structs

/// Top-level SRTP/SRTCP context.  Holds the master keying material and a
/// per-SSRC sub-context for each stream.
///
/// A single `Context` should be used for *either* sending *or* receiving, not
/// both; the replay window and index tracking are not thread-safe.
///
/// TODO: support re-keying via MKI.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Context<P: Protocol> {
    pub master_key: Vec<u8>,
    pub master_salt: Vec<u8>,
    /// Stores `log2(kdr) + 1`, or 0 meaning kdr = 0 (derive once, no re-key).
    /// The actual derivation rate is `2^(key_derivation_rate - 1)`.
    pub key_derivation_rate: u8,
    pub encryption: EncryptionAlgorithm,
    pub authentication: AuthenticationAlgorithm,
    /// Length of the HMAC-SHA1 auth tag appended to each packet (bytes).
    /// Default = 10 (80-bit tag per RFC 3711 §4.2).
    pub auth_tag_len: usize,
    /// Number of as-yet-unknown SSRCs to accept before rejecting new ones.
    pub unknown_ssrcs: usize,
    pub ssrc_context: BTreeMap<u32, SsrcContext<P>>,
}

/// Per-SSRC state: derived session keys and the anti-replay window.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SsrcContext<P: Protocol> {
    /// Highest packet index seen so far (used as replay-window head).
    pub replay_window_head: u64,
    /// Circular bitset; a set bit means that index has been received.
    pub replay_window: FixedBitSet,
    /// 128-bit (16-byte) AES-CM session encryption key.
    pub session_encr_key: Vec<u8>,
    /// 112-bit (14-byte) session salt.
    pub session_salt_key: Vec<u8>,
    /// 160-bit (20-byte) HMAC-SHA1 session authentication key.
    pub session_auth_key: Vec<u8>,
    /// Protocol-specific state (ROC + highest seq for SRTP; index for SRTCP).
    pub protocol_specific: P,
}

// SRTP protocol-specific state

/// SRTP (RTP) per-SSRC state.
/// Tracks the rollover counter and highest sequence number for index estimation.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct Srtp {
    pub rollover_counter: u32,
    pub highest_seq_num: u16,
}

/// SRTCP (RTCP) per-SSRC state.
/// The packet index is carried in-band, so only the highest *sent* index
/// needs to be tracked (for outgoing index assignment).
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct Srtcp {
    /// 31-bit SRTCP index (E-bit is stored separately in the packet).
    pub highest_sent_index: u32,
}

// Srtp helpers

impl Srtp {
    /// Parse the 16-bit RTP sequence number from a raw packet buffer.
    fn parse_seq_num(packet: &[u8]) -> Result<u16> {
        let reader = &mut &packet[..];
        let header = track_try!(rfc3550::RtpFixedHeader::read_from(reader));
        Ok(header.seq_num)
    }

    /// Estimate the 48-bit packet index from the 16-bit sequence number using
    /// the current ROC, minimising the absolute index distance.
    /// Algorithm: RFC 3711 §3.3.1.
    fn estimate_packet_index(state: &Self, seq_num: u16) -> U48 {
        const MID: u16 = 1 << 15;
        let probable_roc = if state.highest_seq_num < MID {
            if state.highest_seq_num.wrapping_add(MID) < seq_num {
                state.rollover_counter.wrapping_sub(1)
            } else {
                state.rollover_counter
            }
        } else if state.highest_seq_num.wrapping_sub(MID) > seq_num {
            state.rollover_counter.wrapping_add(1)
        } else {
            state.rollover_counter
        };
        (U48::from(probable_roc) << 16) + U48::from(seq_num)
    }
}

// Protocol impl for SRTP

impl Protocol for Srtp {
    type PacketIndex = U48;
    const ENC_KEY_LABEL: u8 = 0;
    const AUTH_KEY_LABEL: u8 = 1;
    const SALT_KEY_LABEL: u8 = 2;

    fn read_ssrc(packet: &[u8]) -> Result<Ssrc> {
        let header = track_try!(rfc3550::RtpFixedHeader::read_from(&mut &packet[..]));
        Ok(header.ssrc)
    }

    fn determine_incoming_packet_index(
        _context: &Context<Self>,
        ssrc_context: &SsrcContext<Self>,
        packet: &[u8],
    ) -> Result<Self::PacketIndex> {
        let seq_num = track_try!(Srtp::parse_seq_num(packet));
        Ok(Srtp::estimate_packet_index(&ssrc_context.protocol_specific, seq_num))
    }

    fn determine_outgoing_packet_index(
        context: &SsrcContext<Self>,
        packet: &[u8],
    ) -> Result<Self::PacketIndex> {
        let seq_num = track_try!(Srtp::parse_seq_num(packet));
        Ok(Srtp::estimate_packet_index(&context.protocol_specific, seq_num))
    }

    /// For SRTP the authenticated bytes are: RTP header + payload + ROC.
    /// The ROC is not in the wire packet but is appended here for HMAC input.
    fn get_authenticated_bytes<'a>(
        _context: &Context<Self>,
        index: Self::PacketIndex,
        auth_portion: &'a [u8],
    ) -> Result<Cow<'a, [u8]>> {
        let roc = (index >> 16) as u32;
        let mut auth_bytes = auth_portion.to_vec();
        track_try!((&mut auth_bytes).write_u32be(roc));
        Ok(Cow::Owned(auth_bytes))
    }

    fn decrypt(
        context: &Context<Self>,
        ssrc_context: &SsrcContext<Self>,
        packet: &[u8],
        index: Self::PacketIndex,
    ) -> Result<Vec<u8>> {
        let reader = &mut &packet[..];
        let header = track_try!(rfc3550::RtpFixedHeader::read_from(reader));
        let ssrc = header.ssrc;
        // Everything after the fixed header, minus the trailing auth tag.
        let encrypted_portion = &reader[..reader.len() - context.auth_tag_len];

        let mut decrypted = Vec::new();
        track_try!(header.write_to(&mut decrypted));
        context.apply_keystream(ssrc_context, encrypted_portion, &mut decrypted, ssrc, index);
        Ok(decrypted)
    }

    fn encrypt(
        context: &Context<Self>,
        ssrc_context: &SsrcContext<Self>,
        packet: &[u8],
        index: Self::PacketIndex,
    ) -> Result<Vec<u8>> {
        let reader = &mut &packet[..];
        let header = track_try!(rfc3550::RtpFixedHeader::read_from(reader));
        let ssrc = header.ssrc;
        let plaintext_portion = &reader[..]; // full payload (no auth tag yet)

        let mut encrypted = Vec::new();
        track_try!(header.write_to(&mut encrypted));
        context.apply_keystream(ssrc_context, plaintext_portion, &mut encrypted, ssrc, index);
        Ok(encrypted)
    }

    fn update_highest_recv_index(context: &mut SsrcContext<Self>, index: Self::PacketIndex) {
        // RFC 3711 §3.3.1
        let state = &mut context.protocol_specific;
        let roc = (index >> 16) as u32;
        let seq = index as u16;
        match roc.cmp(&state.rollover_counter) {
            std::cmp::Ordering::Equal => {
                if seq > state.highest_seq_num {
                    state.highest_seq_num = seq;
                }
            }
            std::cmp::Ordering::Greater => {
                state.highest_seq_num = seq;
                state.rollover_counter = roc;
            }
            std::cmp::Ordering::Less => {}
        }
    }

    fn update_highest_sent_index(context: &mut SsrcContext<Self>, index: Self::PacketIndex) {
        // Outgoing packets may arrive out of order at the far end, so use the
        // same ROC-tracking logic as receive.
        Self::update_highest_recv_index(context, index);
    }
}

// Protocol impl for SRTCP

impl Protocol for Srtcp {
    type PacketIndex = u32; // 31-bit index; bit 31 is the E-bit in the packet
    const ENC_KEY_LABEL: u8 = 3;
    const AUTH_KEY_LABEL: u8 = 4;
    const SALT_KEY_LABEL: u8 = 5;

    fn read_ssrc(packet: &[u8]) -> Result<Ssrc> {
        let reader = &mut &packet[..];
        track_try!(reader.read_u32be()); // version/padding/RC/PT/length
        track_err!(reader.read_u32be())  // SSRC
    }

    fn determine_incoming_packet_index(
        context: &Context<Self>,
        _ssrc_context: &SsrcContext<Self>,
        packet: &[u8],
    ) -> Result<Self::PacketIndex> {
        // The SRTCP index is the 32-bit word immediately before the auth tag,
        // with bit 31 (the E-bit) masked off.
        let reader = &mut &packet[packet.len() - context.auth_tag_len - 4..];
        let word = track_try!(reader.read_u32be());
        Ok(word & 0x7FFF_FFFF)
    }

    fn determine_outgoing_packet_index(
        context: &SsrcContext<Self>,
        _packet: &[u8],
    ) -> Result<Self::PacketIndex> {
        // Strictly monotone; wraps at 2^31.
        Ok(context.protocol_specific.highest_sent_index.wrapping_add(1) & 0x7FFF_FFFF)
    }

    /// For SRTCP the authenticated bytes are the entire packet up to (and
    /// including) the SRTCP index word - already in the wire format.
    fn get_authenticated_bytes<'a>(
        _context: &Context<Self>,
        _index: Self::PacketIndex,
        auth_portion: &'a [u8],
    ) -> Result<Cow<'a, [u8]>> {
        Ok(Cow::Borrowed(auth_portion))
    }

    fn decrypt(
        context: &Context<Self>,
        ssrc_context: &SsrcContext<Self>,
        packet: &[u8],
        index: Self::PacketIndex,
    ) -> Result<Vec<u8>> {
        // Check the E-bit; if not set the payload is unencrypted.
        let index_word_offset = packet.len() - context.auth_tag_len - 4;
        let e_bit = {
            let reader = &mut &packet[index_word_offset..];
            track_try!(reader.read_u32be()) & 0x8000_0000 != 0
        };
        if !e_bit {
            return Ok(packet[..index_word_offset].to_vec());
        }

        let reader = &mut &packet[..];
        let _ = track_try!(reader.read_u32be()); // V/P/RC/PT/length
        let ssrc = track_try!(reader.read_u32be());
        // RTCP payload: everything after the first 8 bytes, before index+auth.
        let encrypted_portion = &reader[..reader.len() - context.auth_tag_len - 4];

        let mut decrypted = packet[..8].to_vec();
        context.apply_keystream(ssrc_context, encrypted_portion, &mut decrypted, ssrc, index);
        Ok(decrypted)
    }

    fn encrypt(
        context: &Context<Self>,
        ssrc_context: &SsrcContext<Self>,
        packet: &[u8],
        index: Self::PacketIndex,
    ) -> Result<Vec<u8>> {
        let reader = &mut &packet[..];
        let _ = track_try!(reader.read_u32be()); // V/P/RC/PT/length
        let ssrc = track_try!(reader.read_u32be());
        let plaintext_portion = &reader[..];

        // AES-CM: encryption = decryption (XOR with keystream)
        let mut encrypted = packet[..8].to_vec();
        context.apply_keystream(ssrc_context, plaintext_portion, &mut encrypted, ssrc, index);
        // Append the SRTCP index with the E-bit set.
        track_try!(encrypted.write_u32be(index | 0x8000_0000));
        Ok(encrypted)
    }

    fn update_highest_recv_index(_context: &mut SsrcContext<Self>, _index: Self::PacketIndex) {
        // The full index is carried in-band; no tracking needed for receive.
    }

    fn update_highest_sent_index(context: &mut SsrcContext<Self>, index: Self::PacketIndex) {
        context.protocol_specific.highest_sent_index = index;
    }
}

// Context<P> impl

impl<P: Protocol> Context<P>
where
    u64: From<P::PacketIndex>,
{
    /// Create a new context with the given master key and salt.
    /// All other fields take their RFC-recommended defaults:
    /// AES-CM encryption, HMAC-SHA1 authentication, 80-bit auth tag.
    pub fn new(master_key: &[u8], master_salt: &[u8]) -> Self {
        Self {
            master_key: master_key.to_vec(),
            master_salt: master_salt.to_vec(),
            key_derivation_rate: 0,
            encryption: EncryptionAlgorithm::default(),
            authentication: AuthenticationAlgorithm::default(),
            auth_tag_len: 80 / 8,
            unknown_ssrcs: 0,
            ssrc_context: BTreeMap::new(),
        }
    }

    /// Register a known SSRC, initialising its per-SSRC context.
    /// Panics if the SSRC is already registered.
    pub fn add_ssrc(&mut self, ssrc: Ssrc) {
        let prev = self.ssrc_context.insert(ssrc, SsrcContext::new_empty());
        assert!(prev.is_none(), "SSRC {ssrc} had already been added");
    }

    /// Allow `count` additional unknown SSRCs to be auto-registered on first
    /// receipt.  Useful when the far-end SSRC is not known in advance.
    pub fn add_unknown_ssrcs(&mut self, count: usize) {
        self.unknown_ssrcs += count;
    }

    // Key derivation

    fn packet_index_u64(index: P::PacketIndex) -> u64 {
        u64::from(index)
    }

    fn packet_index_be_48(index: P::PacketIndex) -> [u8; 6] {
        let bytes = Self::packet_index_u64(index).to_be_bytes();
        let mut out = [0u8; 6];
        out.copy_from_slice(&bytes[2..]);
        out
    }

    fn key_derivation_x(label: u8, r: u64, master_salt: &[u8]) -> [u8; 16] {
        let mut x = [0u8; 16];
        let copy_len = master_salt.len().min(14);
        x[..copy_len].copy_from_slice(&master_salt[..copy_len]);
        x[7] ^= label;
        let r_bytes = r.to_be_bytes();
        x[8] ^= r_bytes[2];
        x[9] ^= r_bytes[3];
        x[10] ^= r_bytes[4];
        x[11] ^= r_bytes[5];
        x[12] ^= r_bytes[6];
        x[13] ^= r_bytes[7];
        x
    }

    /// Re-derive the session keys for `ssrc` at packet `index`.
    /// (RFC 3711 §4.3)
    ///
    /// The derivation key `r` is computed from `index` and
    /// `key_derivation_rate`, then the PRF is applied with different labels
    /// to produce three independent session keys.
    pub fn update_session_keys(&mut self, ssrc: Ssrc, index: P::PacketIndex) {
        let r: u64 = if self.key_derivation_rate == 0 {
            0
        } else {
            Self::packet_index_u64(index) >> (self.key_derivation_rate - 1)
        };
        // TODO: cache `r`; skip re-derivation when it has not changed.

        let enc_x = Self::key_derivation_x(P::ENC_KEY_LABEL, r, &self.master_salt);
        let auth_x = Self::key_derivation_x(P::AUTH_KEY_LABEL, r, &self.master_salt);
        let salt_x = Self::key_derivation_x(P::SALT_KEY_LABEL, r, &self.master_salt);

        let ctx = self.ssrc_context.get_mut(&ssrc).unwrap();
        let enc_len = ctx.session_encr_key.len();
        let auth_len = ctx.session_auth_key.len();
        let salt_len = ctx.session_salt_key.len();

        ctx.session_encr_key = prf_n(&self.master_key, &enc_x, enc_len);
        ctx.session_auth_key = prf_n(&self.master_key, &auth_x, auth_len);
        ctx.session_salt_key = prf_n(&self.master_key, &salt_x, salt_len);
    }

    // Authentication

    /// Verify the auth tag on an incoming packet.
    /// `packet` must include the trailing auth tag.
    pub fn authenticate(
        &self,
        ssrc_context: &SsrcContext<P>,
        packet: &[u8],
        index: P::PacketIndex,
    ) -> Result<()> {
        let (body, tag) = packet.split_at(packet.len() - self.auth_tag_len);
        let auth_bytes = track_try!(P::get_authenticated_bytes(self, index, body));
        let mut expected = hmac_hash_sha1(&ssrc_context.session_auth_key, &auth_bytes);
        expected.truncate(self.auth_tag_len);
        track_assert_eq!(tag, expected.as_slice(), ErrorKind::Invalid);
        Ok(())
    }

    /// Compute and return the auth tag for an outgoing packet (without tag).
    pub fn generate_auth_tag(
        &self,
        ssrc_context: &SsrcContext<P>,
        packet: &[u8],
        index: P::PacketIndex,
    ) -> Result<Vec<u8>> {
        let auth_bytes = track_try!(P::get_authenticated_bytes(self, index, packet));
        let mut tag = hmac_hash_sha1(&ssrc_context.session_auth_key, &auth_bytes);
        tag.truncate(self.auth_tag_len);
        Ok(tag)
    }

    // AES-CM core

    /// Build the 16-byte AES-CM IV for packet `index` from SSRC `ssrc`.
    ///
    /// RFC 3711 §4.1:
    ///   IV = (k_s * 2^16) XOR (SSRC * 2^64) XOR (index * 2^16)
    ///
    fn build_iv(ssrc_context: &SsrcContext<P>, ssrc: u32, index: P::PacketIndex) -> [u8; 16] {
        let mut iv = [0u8; 16];
        let salt_len = ssrc_context.session_salt_key.len().min(14);
        iv[..salt_len].copy_from_slice(&ssrc_context.session_salt_key[..salt_len]);

        let ssrc_bytes = ssrc.to_be_bytes();
        for (dst, src) in iv[4..8].iter_mut().zip(ssrc_bytes.iter()) {
            *dst ^= *src;
        }

        let index_bytes = Self::packet_index_be_48(index);
        for (dst, src) in iv[8..14].iter_mut().zip(index_bytes.iter()) {
            *dst ^= *src;
        }

        iv
    }

    /// Apply the AES-CM keystream to `input`, appending results to `output`.
    /// This is identical for encryption and decryption (XOR is its own inverse).
    ///
    /// Input is processed in `session_encr_key.len()`-byte blocks; the final
    /// block may be a partial block (CTR mode is stream-like).
    fn apply_keystream(
        &self,
        ssrc_context: &SsrcContext<P>,
        input: &[u8],
        output: &mut Vec<u8>,
        ssrc: u32,
        index: P::PacketIndex,
    ) {
        let iv = Self::build_iv(ssrc_context, ssrc, index);
        let mut ctr = Ctr128BE::<Aes128>::new_from_slices(&ssrc_context.session_encr_key, &iv)
            .expect("session_encr_key and IV are always 16 bytes");

        let block_size = ssrc_context.session_encr_key.len();
        for block in input.chunks(block_size) {
            let start = output.len();
            output.extend_from_slice(block);
            ctr.apply_keystream(&mut output[start..]);
        }
    }

    // Kept as aliases so existing callers that go through Protocol::decrypt /
    // Protocol::encrypt are not disturbed.
    pub fn decrypt(
        &self,
        ssrc_context: &SsrcContext<P>,
        packet: &[u8],
        index: P::PacketIndex,
    ) -> Result<Vec<u8>> {
        P::decrypt(self, ssrc_context, packet, index)
    }

    pub fn encrypt(
        &self,
        ssrc_context: &SsrcContext<P>,
        packet: &[u8],
        index: P::PacketIndex,
    ) -> Result<Vec<u8>> {
        P::encrypt(self, ssrc_context, packet, index)
    }

    /// Backward-compatible wrapper - calls [`apply_keystream`] internally.
    /// Kept pub so that callers in `rfc5764.rs` do not need updating yet.
    #[inline]
    pub fn decrypt_portion(
        &self,
        ssrc_context: &SsrcContext<P>,
        encrypted: &[u8],
        decrypted: &mut Vec<u8>,
        ssrc: u32,
        index: P::PacketIndex,
    ) {
        self.apply_keystream(ssrc_context, encrypted, decrypted, ssrc, index);
    }

    /// Backward-compatible wrapper - calls [`apply_keystream`] internally.
    /// Kept pub so that callers in `rfc5764.rs` do not need updating yet.
    #[inline]
    pub fn encrypt_portion(
        &self,
        ssrc_context: &SsrcContext<P>,
        plaintext: &[u8],
        encrypted: &mut Vec<u8>,
        ssrc: u32,
        index: P::PacketIndex,
    ) {
        self.apply_keystream(ssrc_context, plaintext, encrypted, ssrc, index);
    }

    // Packet pipeline

    /// Receive pipeline (RFC 3711 §3.3 / §3.4:
    ///
    ///   1. Identify SSRC context (auto-register if `unknown_ssrcs > 0`).
    ///   2. Estimate / read the full packet index.
    ///   3. Derive session keys for this index.
    ///   4. Enforce replay-window - reject duplicates and very old packets.
    ///   5. Verify the HMAC-SHA1 auth tag.
    ///   6. Decrypt the payload.
    ///   7. Update the replay window, ROC, and highest sequence number.
    pub fn process_incoming(&mut self, packet: &[u8]) -> Result<Vec<u8>> {
        let ssrc = track_try!(P::read_ssrc(packet));
        self.ensure_ssrc(ssrc)?;

        let index = track_try!(P::determine_incoming_packet_index(
            self,
            self.ssrc_context.get(&ssrc).unwrap(),
            packet,
        ));

        self.update_session_keys(ssrc, index);

        let idx = u64::from(index);
        let (result, window_size) = {
            let ctx = self.ssrc_context.get(&ssrc).unwrap();
            let window_size = ctx.replay_window.len() as u64;

            // Replay check
            if idx <= ctx.replay_window_head {
                track_assert!(
                    idx + window_size > ctx.replay_window_head,
                    ErrorKind::Invalid
                );
                track_assert!(
                    !ctx.replay_window[(idx % window_size) as usize],
                    ErrorKind::Invalid
                );
            }

            track_try!(self.authenticate(ctx, packet, index));
            let result = track_try!(self.decrypt(ctx, packet, index));
            (result, window_size)
        };

        {
            let ctx = self.ssrc_context.get_mut(&ssrc).unwrap();
            if idx > ctx.replay_window_head {
                // Clear the slots that are sliding out of the window.
                let advance = idx - ctx.replay_window_head;
                if advance >= window_size {
                    ctx.replay_window.clear();
                } else {
                    let start = ((ctx.replay_window_head + 1) % window_size) as usize;
                    let end = (idx % window_size) as usize;
                    if start > end {
                        ctx.replay_window.set_range(start.., false);
                        ctx.replay_window.set_range(..end, false);
                    } else {
                        ctx.replay_window.set_range(start..end, false);
                    }
                }
                ctx.replay_window_head = idx;
            }
            ctx.replay_window.insert((idx % window_size) as usize);
            P::update_highest_recv_index(ctx, index);
        }

        Ok(result)
    }

    /// Send pipeline (RFC 3711 §3.3 / §3.4):
    ///
    ///   1. Identify SSRC context.
    ///   2. Assign the outgoing packet index.
    ///   3. Derive session keys for this index.
    ///   4. Encrypt the payload.
    ///   5. Compute and append the HMAC-SHA1 auth tag.
    ///   6. Update the index state.
    pub fn process_outgoing(&mut self, packet: &[u8]) -> Result<Vec<u8>> {
        let ssrc = track_try!(P::read_ssrc(packet));
        self.ensure_ssrc(ssrc)?;

        let index = track_try!(P::determine_outgoing_packet_index(
            self.ssrc_context.get(&ssrc).unwrap(),
            packet,
        ));

        self.update_session_keys(ssrc, index);

        let mut result = track_try!(
            self.encrypt(self.ssrc_context.get(&ssrc).unwrap(), packet, index)
        );

        // TODO: append MKI if configured
        let tag = track_try!(self.generate_auth_tag(
            self.ssrc_context.get(&ssrc).unwrap(),
            &result,
            index,
        ));
        result.extend_from_slice(&tag);

        P::update_highest_sent_index(self.ssrc_context.get_mut(&ssrc).unwrap(), index);

        Ok(result)
    }

    // Private helpers

    /// Ensure an SSRC context exists, auto-creating it from the `unknown_ssrcs`
    /// budget if necessary.
    fn ensure_ssrc(&mut self, ssrc: Ssrc) -> Result<()> {
        if !self.ssrc_context.contains_key(&ssrc) {
            track_assert!(
                self.unknown_ssrcs > 0,
                ErrorKind::Invalid,
                "Unknown SSRC {ssrc}"
            );
            self.unknown_ssrcs -= 1;
            self.ssrc_context.insert(ssrc, SsrcContext::new_empty());
        }
        Ok(())
    }
}

// SsrcContext helper

impl<P: Protocol> SsrcContext<P> {
    /// Construct a zeroed-out SSRC context with default key/salt/auth sizes.
    fn new_empty() -> Self {
        Self {
            replay_window_head: 0,
            replay_window: FixedBitSet::with_capacity(128),
            session_encr_key: vec![0u8; 16],  // AES-128: 128 bits
            session_salt_key: vec![0u8; 14],  // AES-CM salt: 112 bits
            session_auth_key: vec![0u8; 20],  // HMAC-SHA1: 160 bits
            protocol_specific: P::default(),
        }
    }
}

// Packet reader/writer wrappers
// These are thin newtype wrappers that plug SRTP/SRTCP processing into the
// existing rfc3550 ReadPacket / WritePacket trait infrastructure.

/// Wraps an inner RTP packet reader, transparently decrypting SRTP packets.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SrtpPacketReader<T> {
    context: Context<Srtp>,
    inner: T,
}

impl<T> SrtpPacketReader<T>
where
    T: ReadPacket,
    T::Packet: RtpPacket,
{
    pub fn new(context: Context<Srtp>, inner: T) -> Self {
        Self { context, inner }
    }
}

impl<T> ReadPacket for SrtpPacketReader<T>
where
    T: ReadPacket,
    T::Packet: RtpPacket,
{
    type Packet = T::Packet;
    fn read_packet<R: Read>(&mut self, reader: &mut R) -> Result<Self::Packet> {
        let raw = track_try!(reader.read_all_bytes());
        let plain = track_try!(self.context.process_incoming(&raw));
        track_err!(self.inner.read_packet(&mut plain.as_slice()))
    }
    fn supports_type(&self, ty: u8) -> bool {
        self.inner.supports_type(ty)
    }
}

/// Wraps an inner RTP packet writer, transparently encrypting to SRTP.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SrtpPacketWriter<T> {
    context: Context<Srtp>,
    inner: T,
}

impl<T> SrtpPacketWriter<T>
where
    T: WritePacket,
    T::Packet: RtpPacket,
{
    pub fn new(context: Context<Srtp>, inner: T) -> Self {
        Self { context, inner }
    }
}

impl<T> WritePacket for SrtpPacketWriter<T>
where
    T: WritePacket,
    T::Packet: RtpPacket,
{
    type Packet = T::Packet;
    fn write_packet<W: Write>(&mut self, writer: &mut W, packet: &T::Packet) -> Result<()> {
        let mut raw = Vec::new();
        track_try!(self.inner.write_packet(&mut raw, packet));
        let cipher = track_try!(self.context.process_outgoing(&raw));
        track_err!(writer.write_all(&cipher))
    }
}

/// Wraps an inner RTCP packet reader, transparently decrypting SRTCP packets.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SrtcpPacketReader<T> {
    context: Context<Srtcp>,
    inner: T,
}

impl<T> SrtcpPacketReader<T>
where
    T: ReadPacket,
    T::Packet: RtcpPacket,
{
    pub fn new(context: Context<Srtcp>, inner: T) -> Self {
        Self { context, inner }
    }
}

impl<T> ReadPacket for SrtcpPacketReader<T>
where
    T: ReadPacket,
    T::Packet: RtcpPacket,
{
    type Packet = T::Packet;
    fn read_packet<R: Read>(&mut self, reader: &mut R) -> Result<Self::Packet> {
        let raw = track_try!(reader.read_all_bytes());
        let plain = track_try!(self.context.process_incoming(&raw));
        track_err!(self.inner.read_packet(&mut plain.as_slice()))
    }
    fn supports_type(&self, ty: u8) -> bool {
        self.inner.supports_type(ty)
    }
}

/// Wraps an inner RTCP packet writer, transparently encrypting to SRTCP.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SrtcpPacketWriter<T> {
    context: Context<Srtcp>,
    inner: T,
}

impl<T> SrtcpPacketWriter<T>
where
    T: WritePacket,
    T::Packet: RtpPacket,
{
    pub fn new(context: Context<Srtcp>, inner: T) -> Self {
        Self { context, inner }
    }
}

// FIXME: bound should likely be `T::Packet: RtcpPacket` - verify against rfc5764.rs
impl<T> WritePacket for SrtcpPacketWriter<T>
where
    T: WritePacket,
    T::Packet: RtpPacket,
{
    type Packet = T::Packet;
    fn write_packet<W: Write>(&mut self, writer: &mut W, packet: &T::Packet) -> Result<()> {
        let mut raw = Vec::new();
        track_try!(self.inner.write_packet(&mut raw, packet));
        let cipher = track_try!(self.context.process_outgoing(&raw));
        track_err!(writer.write_all(&cipher))
    }
}

// Crypto primitives

/// Compute HMAC-SHA1 over `data` keyed by `key`.
/// Returns the full 20-byte digest; callers truncate to `auth_tag_len`.
///
/// Uses the fully-qualified `KeyInit` path to avoid ambiguity between
/// `KeyInit::new_from_slice` and `Mac::new_from_slice`.
fn hmac_hash_sha1(key: &[u8], data: &[u8]) -> Vec<u8> {
    let mut mac =
        <HmacSha1 as KeyInit>::new_from_slice(key).expect("HMAC-SHA1 accepts any key length");
    mac.update(data);
    mac.finalize().into_bytes().to_vec()
}

/// AES-CM pseudo-random function (PRF_n) - RFC 3711 §4.3.1.
///
/// Produces `n` bytes by encrypting successive 128-bit counter blocks with
/// AES-128, where block `i` is: AES_k( x XOR i ).
/// `x` is the key-derivation input (label XOR master-salt XOR key-id).
///
/// The counter `i` is a 16-bit value incremented per 16-byte output block.
fn prf_n(master_key: &[u8], x: &[u8; 16], n: usize) -> Vec<u8> {
    let aes = Aes128::new_from_slice(master_key).expect("AES-128 master key must be 16 bytes");
    let mut out = Vec::with_capacity(n);
    let mut counter = u16::from_be_bytes([x[14], x[15]]);

    while out.len() < n {
        let mut block = *x;
        let ctr = counter.to_be_bytes();
        block[14] = ctr[0];
        block[15] = ctr[1];
        aes.encrypt_block((&mut block).into());

        let remaining = n - out.len();
        out.extend_from_slice(&block[..remaining.min(block.len())]);
        counter = counter.wrapping_add(1);
    }

    out
}

// Tests

#[cfg(test)]
pub(crate) mod test {
    use super::*;
    use crate::rfc3550;
    use crate::rfc4585;

    // Shared test vectors

    pub(crate) const TEST_MASTER_KEY: &[u8] = &[
        211, 77, 116, 243, 125, 116, 231, 95, 59, 219, 79, 118, 241, 189, 244, 119,
    ];
    pub(crate) const TEST_MASTER_SALT: &[u8] = &[
        127, 31, 227, 93, 120, 247, 126, 117, 231, 159, 123, 235, 95, 122,
    ];
    pub(crate) const TEST_SRTP_SSRC: Ssrc = 446_919_554;
    pub(crate) static TEST_SRTP_PACKET: &[u8] = &[
        128, 0, 3, 92, 222, 161, 6, 76, 26, 163, 115, 130, 222, 0, 143, 87, 0, 227, 123, 91, 200,
        238, 141, 220, 9, 191, 52, 111, 100, 62, 220, 158, 211, 79, 184, 199, 79, 182, 9, 248, 170,
        82, 125, 152, 143, 206, 8, 152, 80, 207, 27, 183, 141, 77, 33, 60, 101, 180, 210, 146, 139,
        170, 149, 13, 99, 75, 223, 156, 79, 71, 84, 119, 68, 236, 244, 163, 198, 175, 219, 160,
        255, 9, 82, 169, 64, 112, 106, 4, 0, 246, 39, 29, 88, 15, 62, 174, 21, 253, 171, 198, 128,
        61, 23, 43, 143, 255, 176, 125, 223, 23, 188, 90, 103, 139, 223, 56, 162, 35, 27, 225, 117,
        243, 138, 163, 35, 79, 221, 201, 149, 154, 203, 255, 2, 23, 184, 184, 169, 32, 1, 138, 172,
        60, 70, 240, 53, 11, 54, 81, 172, 214, 34, 136, 39, 152, 17, 247, 126, 199, 200, 184, 70,
        7, 52, 191, 129, 239, 86, 78, 172, 229, 178, 112, 22, 125, 191, 164, 17, 193, 24, 152, 197,
        146, 94, 74, 156, 171, 245, 239, 220, 205, 145, 206,
    ];
    pub(crate) const TEST_SRTCP_SSRC: Ssrc = 3_270_675_037;
    pub(crate) static TEST_SRTCP_PACKET: &[u8] = &[
        128, 201, 0, 1, 194, 242, 138, 93, 177, 31, 99, 88, 187, 209, 173, 181, 135, 18, 79, 59,
        119, 153, 115, 34, 75, 94, 96, 29, 32, 14, 118, 86, 145, 159, 203, 174, 225, 34, 196, 229,
        39, 22, 174, 54, 198, 56, 179, 171, 111, 229, 48, 234, 138, 249, 127, 11, 86, 94, 40, 213,
        87, 203, 60, 54, 52, 60, 10, 93, 128, 0, 0, 1, 114, 135, 74, 73, 233, 100, 85, 240, 125,
        93,
    ];

    const TEST_2_MASTER_KEY: &[u8] = &[
        124, 185, 61, 185, 219, 148, 249, 33, 222, 227, 189, 112, 23, 80, 114, 233,
    ];
    const TEST_2_MASTER_SALT: &[u8] =
        &[93, 4, 23, 245, 147, 199, 112, 49, 24, 105, 140, 1, 77, 98];
    const TEST_2_SRTP_SSRC: Ssrc = 180_601_533;
    static TEST_2_SRTP_PACKET_BEFORE_ROLLOVER: &[u8] = &[
        0x80, 0x61, 0xff, 0xff, 0x87, 0xf5, 0xee, 0x93, 0x0a, 0xc3, 0xc2, 0xbd, 0x93, 0x04, 0x0b,
        0x4d, 0xe9, 0x55, 0x69, 0xb7, 0xac, 0x88, 0xc5, 0xd6, 0xc2, 0x75, 0xb8, 0x15, 0x86, 0xc3,
        0xb2, 0x2a, 0x34, 0x64, 0xbe, 0x8b, 0x0d, 0x61, 0xfc, 0x22, 0xf1, 0x30, 0x66, 0xe0, 0x1e,
        0x1d, 0x0c, 0xec, 0xff, 0x8d, 0xff, 0x86, 0xf7, 0xf4, 0x7e, 0x40, 0x8a, 0xd0, 0x36, 0x3f,
        0x67, 0x60, 0x0f, 0xbd, 0x46, 0xa9, 0x3e, 0xa5, 0x4b, 0x31, 0x54, 0xc8, 0x45, 0x61, 0xc8,
        0x33, 0x68, 0x2b, 0x0c, 0x98, 0x5f, 0x61, 0x68, 0xc4, 0x32, 0x8f, 0x70, 0xc4, 0xc6, 0x05,
        0x7e, 0x30, 0xcf, 0x67, 0x78, 0xf4, 0x50, 0x1b, 0xba, 0x5f, 0x10, 0x5f, 0xf6, 0x6b, 0x99,
        0x6d, 0x68, 0xb8, 0x87, 0x21, 0x46, 0xd1, 0x4a, 0x4a,
    ];
    static TEST_2_SRTP_PACKET_AFTER_ROLLOVER: &[u8] = &[
        128, 97, 0, 0, 135, 245, 242, 83, 10, 195, 194, 189, 254, 253, 61, 217, 224, 102, 52, 18,
        244, 100, 144, 73, 190, 225, 100, 195, 28, 35, 116, 15, 37, 91, 236, 28, 24, 134, 223, 188,
        129, 1, 164, 18, 143, 87, 6, 25, 195, 159, 33, 147, 36, 175, 190, 60, 215, 204, 240, 27,
        186, 247, 223, 217, 65, 189, 66, 59, 3, 214, 53, 146, 32, 234, 27, 127, 211, 58, 156, 25,
        139, 236, 11, 138, 245, 134, 84, 164, 130, 226, 90, 74, 131, 57, 100, 0, 106, 127, 239,
        184, 235, 197, 164, 15, 233, 146, 84, 127, 42, 9, 100,
    ];

    // Tests

    #[test]
    fn rtp_packet_index_estimation_works() {
        let mut state = Srtp::default();
        let roc = 0u32;
        let roc_n1 = roc.wrapping_sub(1);
        let roc_p1 = roc.wrapping_add(1);
        state.rollover_counter = roc;

        let idx = |r: u32, s: u16| (u64::from(r) << 16) | u64::from(s);
        let est = |s: &Srtp, seq| Srtp::estimate_packet_index(s, seq);

        state.highest_seq_num = 1000;
        assert_eq!(est(&state, 1), idx(roc, 1));
        assert_eq!(est(&state, 10001), idx(roc, 10001));
        assert_eq!(est(&state, 60001), idx(roc_n1, 60001));

        state.highest_seq_num = 60000;
        assert_eq!(est(&state, 60001), idx(roc, 60001));
        assert_eq!(est(&state, 30001), idx(roc, 30001));
        assert_eq!(est(&state, 10001), idx(roc_p1, 10001));
    }

    #[test]
    fn rtp_decryption_works() {
        let mut context = Context::new(TEST_MASTER_KEY, TEST_MASTER_SALT);
        context.add_ssrc(TEST_SRTP_SSRC);
        let mut reader = SrtpPacketReader::new(context, rfc3550::RtpPacketReader);
        let mut packet_bytes = TEST_SRTP_PACKET;
        let packet = reader.read_packet(&mut packet_bytes).unwrap();

        let expected_prefix = [
            0xbe, 0x9c, 0x8c, 0x86, 0x81, 0x80, 0x81, 0x86, 0x8d, 0x9c, 0xfd, 0x1b, 0x0d, 0x05,
            0x01, 0x00, 0x01, 0x05, 0x0d, 0x1b, 0xff, 0x9b, 0x8d, 0x85, 0x81, 0x80, 0x81, 0x85,
            0x8d, 0x9b, 0xff, 0x1b,
        ];
        assert_eq!(&packet.payload[..expected_prefix.len()], &expected_prefix);
    }

    #[test]
    fn rtp_decryption_with_rollover_works() {
        let mut context = Context::<Srtp>::new(TEST_2_MASTER_KEY, TEST_2_MASTER_SALT);
        context.add_ssrc(TEST_2_SRTP_SSRC);
        context
            .ssrc_context
            .get_mut(&TEST_2_SRTP_SSRC)
            .unwrap()
            .protocol_specific
            .highest_seq_num = 65534;
        let mut reader = SrtpPacketReader::new(context, rfc3550::RtpPacketReader);
        let mut before = TEST_2_SRTP_PACKET_BEFORE_ROLLOVER;
        let mut after  = TEST_2_SRTP_PACKET_AFTER_ROLLOVER;
        reader.read_packet(&mut before).unwrap();
        reader.read_packet(&mut after).unwrap();
    }

    #[test]
    fn rtcp_decryption_works() {
        let master_key = [
            254, 123, 44, 240, 174, 252, 53, 54, 2, 213, 123, 106, 85, 165, 5, 13,
        ];
        let master_salt = [77, 202, 202, 112, 81, 101, 219, 232, 143, 131, 160, 89, 15, 141];
        let packet = [
            128, 201, 0, 1, 194, 242, 138, 93, 67, 38, 193, 233, 60, 78, 188, 195, 230, 90, 19,
            196, 152, 235, 136, 164, 15, 177, 174, 217, 207, 115, 148, 223, 109, 112, 71, 245, 16,
            214, 216, 232, 87, 153, 5, 238, 72, 201, 223, 43, 69, 99, 54, 211, 118, 28, 227, 100,
            161, 216, 90, 203, 99, 167, 215, 130, 151, 16, 128, 138, 128, 0, 0, 1, 126, 39, 201,
            236, 161, 194, 6, 232, 194, 230,
        ];
        let mut context = Context::new(&master_key, &master_salt);
        context.add_ssrc(TEST_SRTCP_SSRC);
        let mut reader = SrtcpPacketReader::new(context, rfc4585::RtcpPacketReader);
        let pkt = track_try_unwrap!(reader.read_packet(&mut &packet[..]));
        println!("# {pkt:?}");
    }

    #[test]
    fn rtp_decryption_encryption_are_inverse() {
        let mut dec = Context::<Srtp>::new(TEST_MASTER_KEY, TEST_MASTER_SALT);
        let mut enc = Context::<Srtp>::new(TEST_MASTER_KEY, TEST_MASTER_SALT);
        dec.add_ssrc(TEST_SRTP_SSRC);
        enc.add_ssrc(TEST_SRTP_SSRC);
        let plain = track_try_unwrap!(dec.process_incoming(TEST_SRTP_PACKET));
        let cipher = track_try_unwrap!(enc.process_outgoing(&plain));
        assert_eq!(cipher.as_slice(), TEST_SRTP_PACKET);
    }

    #[test]
    fn rtcp_decryption_encryption_are_inverse() {
        let mut dec = Context::<Srtcp>::new(TEST_MASTER_KEY, TEST_MASTER_SALT);
        let mut enc = Context::<Srtcp>::new(TEST_MASTER_KEY, TEST_MASTER_SALT);
        dec.add_ssrc(TEST_SRTCP_SSRC);
        enc.add_ssrc(TEST_SRTCP_SSRC);
        let plain = track_try_unwrap!(dec.process_incoming(TEST_SRTCP_PACKET));
        let cipher = track_try_unwrap!(enc.process_outgoing(&plain));
        assert_eq!(cipher.as_slice(), TEST_SRTCP_PACKET);
    }

    #[test]
    fn rtcp_encryption_does_not_use_two_time_pad() {
        let mut dec = Context::<Srtcp>::new(TEST_MASTER_KEY, TEST_MASTER_SALT);
        let mut enc = Context::<Srtcp>::new(TEST_MASTER_KEY, TEST_MASTER_SALT);
        dec.add_ssrc(TEST_SRTCP_SSRC);
        enc.add_ssrc(TEST_SRTCP_SSRC);
        let plain = track_try_unwrap!(dec.process_incoming(TEST_SRTCP_PACKET));
        let c1 = track_try_unwrap!(enc.process_outgoing(&plain));
        let c2 = track_try_unwrap!(enc.process_outgoing(&plain));
        let c3 = track_try_unwrap!(enc.process_outgoing(&plain));
        assert_ne!(c1, c2);
        assert_ne!(c1, c3);
        assert_ne!(c2, c3);
    }

    #[test]
    fn rtp_does_not_allow_packet_replay() {
        let mut ctx = Context::<Srtp>::new(TEST_MASTER_KEY, TEST_MASTER_SALT);
        ctx.add_ssrc(TEST_SRTP_SSRC);
        assert!(ctx.process_incoming(TEST_SRTP_PACKET).is_ok());
        assert!(ctx.process_incoming(TEST_SRTP_PACKET).is_err());
        assert!(ctx.process_incoming(TEST_SRTP_PACKET).is_err());
    }

    #[test]
    fn rtcp_does_not_allow_packet_replay() {
        let mut ctx = Context::<Srtcp>::new(TEST_MASTER_KEY, TEST_MASTER_SALT);
        ctx.add_ssrc(TEST_SRTCP_SSRC);
        assert!(ctx.process_incoming(TEST_SRTCP_PACKET).is_ok());
        assert!(ctx.process_incoming(TEST_SRTCP_PACKET).is_err());
        assert!(ctx.process_incoming(TEST_SRTCP_PACKET).is_err());
    }

    #[test]
    fn rtcp_does_not_allow_delayed_packet_replay() {
        let mut dec = Context::<Srtcp>::new(TEST_MASTER_KEY, TEST_MASTER_SALT);
        dec.add_ssrc(TEST_SRTCP_SSRC);
        let plain = dec.process_incoming(TEST_SRTCP_PACKET).unwrap();

        let mut enc = Context::<Srtcp>::new(TEST_MASTER_KEY, TEST_MASTER_SALT);
        enc.add_ssrc(TEST_SRTCP_SSRC);
        const N: usize = 10;
        let packets: Vec<_> = (0..N)
            .map(|_| enc.process_outgoing(&plain).unwrap())
            .collect();

        let mut dec = Context::<Srtcp>::new(TEST_MASTER_KEY, TEST_MASTER_SALT);
        dec.add_ssrc(TEST_SRTCP_SSRC);
        dec.ssrc_context
            .get_mut(&TEST_SRTCP_SSRC)
            .unwrap()
            .replay_window = FixedBitSet::with_capacity(4);

        for p in &packets[..6] {
            assert!(dec.process_incoming(p).is_ok());
        }
        for p in &packets[..6] {
            assert!(dec.process_incoming(p).is_err());
        }
        assert!(dec.process_incoming(&packets[7]).is_ok());
        assert!(dec.process_incoming(&packets[8]).is_ok());
        assert!(dec.process_incoming(&packets[9]).is_ok());
        assert!(dec.process_incoming(&packets[6]).is_ok());
        assert!(dec.process_incoming(&packets[7]).is_err());
        assert!(dec.process_incoming(&packets[8]).is_err());
        assert!(dec.process_incoming(&packets[9]).is_err());
        assert!(dec.process_incoming(&packets[6]).is_err());
    }
}
