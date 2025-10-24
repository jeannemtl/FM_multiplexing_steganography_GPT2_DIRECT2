"""
DIRECT FDM+iMEC ENCODER (12-bit blocks, optimized for clear FFT)

Pipeline:
1. Generate ASK-modulated carriers for each agent (ALICE, BOB, CHARLIE)
2. Superpose signals (Frequency Division Multiplexing)
3. Quantize continuous signal to bits
4. Encrypt with one-time pad (makes distribution uniform - CRITICAL for iMEC!)
5. Apply iMEC encoding for perfect security

Output: Saves all data needed for decoding to 'encoded_data.pkl'
"""

import numpy as np
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import pickle
import sys
import os

# Disable HF transfer if causing issues
os.environ['HF_HUB_ENABLE_HF_TRANSFER'] = '0'

# Import iMEC encoder
try:
    from imec_encoder import MinEntropyCouplingSteganography
except ImportError:
    print("ERROR: imec_encoder.py not found!")
    print("Please ensure imec_encoder.py is in the same directory.")
    sys.exit(1)


def ask_modulate(bits, freq, length, sample_rate=100.0):
    """
    ASK (Amplitude Shift Keying) modulation with proper sampling.
    
    Args:
        bits: Binary message [0, 1, 0, 1, ...]
        freq: Carrier frequency (Hz)
        length: Total number of samples
        sample_rate: Samples per second
    
    Returns:
        signal: ASK-modulated carrier
    """
    samples_per_bit = length // len(bits)
    signal = np.zeros(length)
    
    for i, bit in enumerate(bits):
        start = i * samples_per_bit
        end = min((i + 1) * samples_per_bit, length)
        
        # Amplitude depends on bit value
        amplitude = 1.0 if bit == 1 else 0.1
        
        # Generate sinusoidal carrier with proper time vector
        # Time duration for this bit segment
        duration = (end - start) / sample_rate
        t = np.linspace(0, duration, end - start)
        carrier = np.sin(2 * np.pi * freq * t)
        
        signal[start:end] = amplitude * carrier
    
    return signal


def quantize_to_bits(signal, bits_per_sample=8):
    """
    Quantize continuous signal to binary representation.
    
    Args:
        signal: Array of continuous amplitude values
        bits_per_sample: Bits per sample (8 = 256 quantization levels)
    
    Returns:
        signal_bits: Binary string
        metadata: Dict with reconstruction parameters
    """
    # Normalize to [0, 1]
    signal_min = float(signal.min())
    signal_max = float(signal.max())
    signal_range = signal_max - signal_min
    
    if signal_range < 1e-10:
        signal_range = 1.0
        print("WARNING: Signal has near-zero range!")
    
    normalized = (signal - signal_min) / signal_range
    
    # Quantize to integer levels
    max_level = (2 ** bits_per_sample) - 1
    quantized = np.round(normalized * max_level).astype(np.uint16)
    
    # Convert to binary string
    signal_bits = ''.join(format(int(val), f'0{bits_per_sample}b') 
                          for val in quantized)
    
    # Save metadata for decoder
    metadata = {
        'num_samples': len(signal),
        'bits_per_sample': bits_per_sample,
        'signal_min': signal_min,
        'signal_max': signal_max,
        'quantization_levels': max_level + 1
    }
    
    return signal_bits, metadata


def encrypt_with_one_time_pad(plaintext_bits):
    """
    Encrypt with one-time pad to create UNIFORM distribution.
    
    ⚠️ CRITICAL: iMEC requires uniform input distribution!
    
    Args:
        plaintext_bits: Binary string
    
    Returns:
        ciphertext_bits: Encrypted binary string (uniform distribution)
        one_time_pad_key: Encryption key (must be kept secret and shared with decoder)
    """
    # Generate random one-time pad
    one_time_pad_key = np.random.randint(0, 2, len(plaintext_bits), dtype=np.uint8)
    
    # XOR encryption
    ciphertext_bits = ''.join(
        str(int(plaintext_bits[i]) ^ int(one_time_pad_key[i]))
        for i in range(len(plaintext_bits))
    )
    
    return ciphertext_bits, one_time_pad_key


def verify_uniformity(bit_string, name="Bit string"):
    """Check if bit string has uniform distribution (should be ~50% ones)."""
    ones_count = sum(int(b) for b in bit_string)
    ones_ratio = ones_count / len(bit_string)
    
    print(f"\n{name} uniformity check:")
    print(f"  Length: {len(bit_string)} bits")
    print(f"  Ones ratio: {ones_ratio:.4f} (target: 0.5000)")
    
    if 0.48 <= ones_ratio <= 0.52:
        print(f"  ✓ Distribution is uniform")
    else:
        print(f"  ⚠️  WARNING: Distribution may not be uniform!")
    
    return ones_ratio


# ============================================================================
# MAIN ENCODING PIPELINE
# ============================================================================

def main():
    print("="*80)
    print("DIRECT FDM+iMEC ENCODER (12-bit blocks)")
    print("="*80)
    
    # ========================================================================
    # CONFIGURATION
    # ========================================================================
    
    # Messages for each agent (16 bits each)
    messages = {
        'ALICE':   [0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 1, 1, 0, 0, 0, 0],
        'BOB':     [1, 1, 1, 0, 1, 0, 1, 1, 0, 1, 1, 0, 1, 0, 0, 0],
        'CHARLIE': [1, 1, 1, 0, 1, 0, 0, 0, 1, 0, 0, 1, 1, 1, 0, 0]
    }
    
    # Carrier frequencies (Hz)
    agent_frequencies = {
        'ALICE': 1.0,
        'BOB': 2.0,
        'CHARLIE': 3.0
    }
    
    # Signal parameters - OPTIMIZED for clear FFT peaks
    num_samples = 5000  # 5x longer for ~3 cycles per bit
    sample_rate = 100.0  # Samples per second
    bits_per_sample = 8  # 256 quantization levels
    
    # GPT-2 context for stegotext
    context = "The future of artificial intelligence"
    
    print(f"\nConfiguration:")
    print(f"  Agents: {list(messages.keys())}")
    print(f"  Message length: {len(messages['ALICE'])} bits per agent")
    print(f"  Signal length: {num_samples} samples")
    print(f"  Sample rate: {sample_rate} Hz")
    print(f"  Samples per bit: {num_samples / len(messages['ALICE']):.1f}")
    print(f"  Cycles per bit @ 1Hz: {(num_samples / len(messages['ALICE'])) / sample_rate:.2f}")
    print(f"  Quantization: {bits_per_sample} bits/sample ({2**bits_per_sample} levels)")
    print(f"  Total bits: {num_samples * bits_per_sample}")
    
    # ========================================================================
    # STEP 1: FREQUENCY MULTIPLEXING
    # ========================================================================
    
    print("\n" + "="*80)
    print("STEP 1: FREQUENCY MULTIPLEXING (ASK + SUPERPOSITION)")
    print("="*80)
    
    alice_signal = ask_modulate(messages['ALICE'], agent_frequencies['ALICE'], 
                                num_samples, sample_rate)
    bob_signal = ask_modulate(messages['BOB'], agent_frequencies['BOB'], 
                              num_samples, sample_rate)
    charlie_signal = ask_modulate(messages['CHARLIE'], agent_frequencies['CHARLIE'], 
                                  num_samples, sample_rate)
    
    # Superpose (FDM)
    combined_signal = alice_signal + bob_signal + charlie_signal
    
    print(f"\nGenerated signals:")
    print(f"  ALICE:   {len(alice_signal)} samples @ {agent_frequencies['ALICE']} Hz")
    print(f"  BOB:     {len(bob_signal)} samples @ {agent_frequencies['BOB']} Hz")
    print(f"  CHARLIE: {len(charlie_signal)} samples @ {agent_frequencies['CHARLIE']} Hz")
    
    print(f"\nCombined signal (superposition):")
    print(f"  Samples: {len(combined_signal)}")
    print(f"  Range: [{combined_signal.min():.3f}, {combined_signal.max():.3f}]")
    print(f"  Mean: {combined_signal.mean():.3f}")
    print(f"  Std: {combined_signal.std():.3f}")
    print(f"  ✓ Superposition creates Gaussian-like distribution")
    
    # ========================================================================
    # STEP 2: QUANTIZATION
    # ========================================================================
    
    print("\n" + "="*80)
    print("STEP 2: QUANTIZATION (SIGNAL → BITS)")
    print("="*80)
    
    signal_bits, quant_metadata = quantize_to_bits(combined_signal, bits_per_sample)
    
    print(f"\n✓ Quantization complete:")
    print(f"  Input: {num_samples} samples")
    print(f"  Output: {len(signal_bits)} bits")
    print(f"  Bits per sample: {bits_per_sample}")
    print(f"  Quantization levels: {quant_metadata['quantization_levels']}")
    
    # ========================================================================
    # STEP 3: ENCRYPTION (ONE-TIME PAD)
    # ========================================================================
    
    print("\n" + "="*80)
    print("STEP 3: ENCRYPTION (ONE-TIME PAD)")
    print("="*80)
    print("⚠️  CRITICAL: Making distribution uniform for iMEC!")
    
    encrypted_bits, one_time_pad_key = encrypt_with_one_time_pad(signal_bits)
    
    print(f"\n✓ Encryption complete:")
    print(f"  Plaintext: {len(signal_bits)} bits")
    print(f"  Ciphertext: {len(encrypted_bits)} bits")
    print(f"  Key: {len(one_time_pad_key)} bits")
    
    verify_uniformity(encrypted_bits, "Ciphertext")
    
    # ========================================================================
    # STEP 4: iMEC ENCODING (12-BIT BLOCKS)
    # ========================================================================
    
    print("\n" + "="*80)
    print("STEP 4: iMEC ENCODING (PERFECT SECURITY) - 12-BIT BLOCKS")
    print("="*80)
    
    print("\nLoading GPT-2...")
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    model = GPT2LMHeadModel.from_pretrained('gpt2')
    model.eval()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    print(f"✓ GPT-2 loaded on {device}")
    
    print("\nInitializing iMEC encoder with 12-bit blocks...")
    imec = MinEntropyCouplingSteganography(block_size_bits=12)
    print(f"✓ iMEC initialized (4,096 values per block)")
    
    # CRITICAL: Use 12-bit blocks
    n_blocks = len(encrypted_bits) // 12
    print(f"\nEncoding {len(encrypted_bits)} bits ({n_blocks} blocks) with iMEC...")
    
    imec_tokens = imec.encode_imec(
        encrypted_bits,
        context,
        max_tokens=10000,
        entropy_threshold=0.05
    )
    
    print(f"\n✓ iMEC encoding complete:")
    print(f"  Input: {len(encrypted_bits)} bits")
    print(f"  Output: {len(imec_tokens)} tokens")
    print(f"  Efficiency: {len(encrypted_bits) / len(imec_tokens):.2f} bits/token")
    
    stegotext = tokenizer.decode(imec_tokens)
    print(f"\nStegotext preview (first 200 chars):")
    print(f"  {stegotext[:200]}...")
    
    # ========================================================================
    # SAVE ENCODED DATA
    # ========================================================================
    
    print("\n" + "="*80)
    print("SAVING ENCODED DATA")
    print("="*80)
    
    encoded_data = {
        # Original messages (for verification)
        'messages': messages,
        'agent_frequencies': agent_frequencies,
        
        # Signal processing parameters
        'num_samples': num_samples,
        'sample_rate': sample_rate,
        'quant_metadata': quant_metadata,
        
        # Encoded data
        'context': context,
        'imec_tokens': imec_tokens,
        'stegotext': stegotext,
        
        # Decryption key (must be kept secret!)
        'one_time_pad_key': one_time_pad_key,
        
        # iMEC parameters (CRITICAL: 12-bit blocks!)
        'block_size_bits': 12,
        
        # For verification
        'combined_signal': combined_signal,
        'individual_signals': {
            'ALICE': alice_signal,
            'BOB': bob_signal,
            'CHARLIE': charlie_signal
        }
    }
    
    output_file = 'encoded_data.pkl'
    with open(output_file, 'wb') as f:
        pickle.dump(encoded_data, f)
    
    print(f"\n✓ Saved to: {output_file}")
    print(f"\nData saved:")
    print(f"  - Stegotext: {len(imec_tokens)} tokens")
    print(f"  - One-time pad key: {len(one_time_pad_key)} bits")
    print(f"  - Block size: 12 bits")
    print(f"  - Metadata: frequencies, quantization params, etc.")
    
    # ========================================================================
    # SUMMARY
    # ========================================================================
    
    print("\n" + "="*80)
    print("✓ ENCODING COMPLETE!")
    print("="*80)
    
    print(f"\nPipeline summary:")
    print(f"  1. Messages:       3 agents × {len(messages['ALICE'])} bits")
    print(f"  2. FDM:            {num_samples} samples (superposition)")
    print(f"  3. Quantization:   {len(signal_bits)} bits")
    print(f"  4. Encryption:     {len(encrypted_bits)} bits (uniform)")
    print(f"  5. iMEC (12-bit):  {len(imec_tokens)} secure tokens")
    
    print(f"\nSignal quality metrics:")
    print(f"  - Samples per bit: {num_samples / len(messages['ALICE']):.1f}")
    print(f"  - Cycles/bit @ 1Hz: {(num_samples / len(messages['ALICE'])) / sample_rate:.2f}")
    print(f"  - Expected FFT peaks: 1.0, 2.0, 3.0 Hz")
    
    print(f"\nNext steps:")
    print(f"  1. Transfer 'encoded_data.pkl' to receiver")
    print(f"  2. Run decoder: python decoder_direct_fdm_imec_12bit.py")
    print(f"  3. Check fft_analysis.png for clear frequency peaks!")
    
    print("\n" + "="*80)


if __name__ == "__main__":
    main()
