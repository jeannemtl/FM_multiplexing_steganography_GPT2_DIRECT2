"""
DIRECT FDM+iMEC DECODER (12-bit blocks)

Pipeline:
1. Load encoded data (stegotext tokens, one-time pad key, metadata)
2. Decode iMEC tokens → encrypted bits
3. Decrypt with one-time pad → signal bits
4. Reconstruct continuous signal from bits
5. Apply FFT + bandpass filtering to extract each agent's message

Input: Reads 'encoded_data.pkl'
Output: Recovered messages for ALICE, BOB, CHARLIE with accuracy metrics
"""

import numpy as np
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import pickle
import sys
from scipy import signal as scipy_signal
from scipy.fft import fft, fftfreq
import matplotlib.pyplot as plt

# Import iMEC encoder
try:
    from imec_encoder import MinEntropyCouplingSteganography
except ImportError:
    print("ERROR: imec_encoder.py not found!")
    print("Please ensure imec_encoder.py is in the same directory.")
    sys.exit(1)


def reconstruct_from_bits(signal_bits, metadata):
    """
    Reconstruct continuous signal from binary representation.
    
    Args:
        signal_bits: Binary string
        metadata: Dict with quantization parameters
    
    Returns:
        reconstructed_signal: Array of continuous amplitude values
    """
    bits_per_sample = metadata['bits_per_sample']
    num_samples = metadata['num_samples']
    
    # Parse bits back to quantized integer values
    quantized = []
    for i in range(num_samples):
        start = i * bits_per_sample
        end = start + bits_per_sample
        
        if end <= len(signal_bits):
            sample_bits = signal_bits[start:end]
            quantized.append(int(sample_bits, 2))
        else:
            print(f"WARNING: Insufficient bits for sample {i}")
            break
    
    quantized = np.array(quantized, dtype=np.float64)
    
    # De-normalize from [0, max_level] to [signal_min, signal_max]
    max_level = metadata['quantization_levels'] - 1
    normalized = quantized / max_level
    
    signal_min = metadata['signal_min']
    signal_max = metadata['signal_max']
    reconstructed = normalized * (signal_max - signal_min) + signal_min
    
    return reconstructed


def bandpass_filter(signal, freq_low, freq_high, sample_rate=100.0, order=4):
    """
    Apply bandpass filter to extract specific frequency band.
    
    Args:
        signal: Input signal
        freq_low: Lower cutoff frequency (Hz)
        freq_high: Upper cutoff frequency (Hz)
        sample_rate: Sampling rate
        order: Filter order
    
    Returns:
        filtered_signal: Bandpass filtered signal
    """
    nyquist = sample_rate * 0.5
    
    # Normalize frequencies to Nyquist frequency
    low = freq_low / nyquist
    high = freq_high / nyquist
    
    # Clamp to valid range (0, 1)
    low = max(min(low, 0.99), 0.01)
    high = max(min(high, 0.99), 0.01)
    
    if low >= high:
        print(f"WARNING: Invalid frequency range [{low}, {high}]")
        return signal
    
    try:
        # Design Butterworth bandpass filter
        b, a = scipy_signal.butter(order, [low, high], btype='band')
        
        # Apply filter (zero-phase filtering)
        filtered = scipy_signal.filtfilt(b, a, signal)
        
        return filtered
    except Exception as e:
        print(f"WARNING: Bandpass filter failed: {e}")
        return signal


def decode_ask(filtered_signal, original_bits_length, sample_rate=100.0):
    """
    Decode ASK-modulated signal to recover binary message.
    
    Args:
        filtered_signal: Bandpass-filtered signal
        original_bits_length: Number of bits to recover
        sample_rate: Sampling rate
    
    Returns:
        recovered_bits: List of decoded bits [0, 1, 0, ...]
    """
    # Envelope detection using Hilbert transform
    analytic_signal = scipy_signal.hilbert(filtered_signal)
    envelope = np.abs(analytic_signal)
    
    # Decode bits from envelope
    samples_per_bit = len(filtered_signal) // original_bits_length
    recovered_bits = []
    
    # Adaptive threshold (between min and max envelope)
    threshold = (envelope.max() + envelope.min()) / 2
    
    for i in range(original_bits_length):
        start = i * samples_per_bit
        end = min((i + 1) * samples_per_bit, len(envelope))
        
        # Average envelope in this bit period
        avg_amplitude = envelope[start:end].mean()
        
        # Threshold decision
        bit = 1 if avg_amplitude > threshold else 0
        recovered_bits.append(bit)
    
    return recovered_bits


def calculate_accuracy(original_bits, recovered_bits):
    """Calculate bit error rate."""
    if len(original_bits) != len(recovered_bits):
        print(f"WARNING: Length mismatch: {len(original_bits)} vs {len(recovered_bits)}")
        min_len = min(len(original_bits), len(recovered_bits))
        original_bits = original_bits[:min_len]
        recovered_bits = recovered_bits[:min_len]
    
    correct = sum(1 for orig, recv in zip(original_bits, recovered_bits) 
                  if orig == recv)
    accuracy = correct / len(original_bits)
    
    return accuracy


def plot_fft_analysis(signal, sample_rate, title="FFT Analysis"):
    """
    Plot FFT to visualize frequency content.
    
    Args:
        signal: Time-domain signal
        sample_rate: Sampling rate
        title: Plot title
    """
    # Compute FFT
    N = len(signal)
    yf = fft(signal)
    xf = fftfreq(N, 1/sample_rate)
    
    # Plot only positive frequencies
    plt.figure(figsize=(12, 4))
    plt.plot(xf[:N//2], 2.0/N * np.abs(yf[:N//2]))
    plt.grid(True)
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Magnitude')
    plt.title(title)
    plt.xlim([0, 5])
    
    # Mark expected carrier frequencies
    for freq, label in [(1.0, 'ALICE'), (2.0, 'BOB'), (3.0, 'CHARLIE')]:
        plt.axvline(x=freq, color='r', linestyle='--', alpha=0.5, label=label)
    
    plt.legend()
    plt.tight_layout()
    plt.savefig('fft_analysis.png', dpi=150)
    print(f"✓ Saved FFT plot to: fft_analysis.png")


# ============================================================================
# MAIN DECODING PIPELINE
# ============================================================================

def main():
    print("="*80)
    print("DIRECT FDM+iMEC DECODER (12-bit blocks)")
    print("="*80)
    
    # ========================================================================
    # LOAD ENCODED DATA
    # ========================================================================
    
    print("\n" + "="*80)
    print("LOADING ENCODED DATA")
    print("="*80)
    
    input_file = 'encoded_data.pkl'
    try:
        with open(input_file, 'rb') as f:
            encoded_data = pickle.load(f)
        print(f"✓ Loaded: {input_file}")
    except FileNotFoundError:
        print(f"ERROR: {input_file} not found!")
        print("Please run encoder_direct_fdm_imec.py first.")
        sys.exit(1)
    
    # Extract data
    context = encoded_data['context']
    imec_tokens = encoded_data['imec_tokens']
    one_time_pad_key = encoded_data['one_time_pad_key']
    quant_metadata = encoded_data['quant_metadata']
    messages = encoded_data['messages']
    agent_frequencies = encoded_data['agent_frequencies']
    sample_rate = encoded_data.get('sample_rate', 100.0)
    
    # CRITICAL: Calculate n_blocks from actual bit length and block size
    block_size_bits = 12  # Must match encoder!
    total_bits = len(one_time_pad_key)
    n_blocks = (total_bits + block_size_bits - 1) // block_size_bits  # Ceiling division
    
    print(f"\nLoaded data:")
    print(f"  Stegotext: {len(imec_tokens)} tokens")
    print(f"  One-time pad: {len(one_time_pad_key)} bits")
    print(f"  Block size: {block_size_bits} bits")
    print(f"  Calculated blocks: {n_blocks}")
    print(f"  Expected decode: {n_blocks * block_size_bits} bits")
    print(f"  Agents: {list(messages.keys())}")
    
    # ========================================================================
    # STEP 1: iMEC DECODING
    # ========================================================================
    
    print("\n" + "="*80)
    print("STEP 1: iMEC DECODING")
    print("="*80)
    
    print("\nLoading GPT-2...")
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    model = GPT2LMHeadModel.from_pretrained('gpt2')
    model.eval()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    print(f"✓ GPT-2 loaded on {device}")
    
    print("\nInitializing iMEC decoder...")
    imec = MinEntropyCouplingSteganography(block_size_bits=12)
    print(f"✓ iMEC initialized (12-bit blocks)")
    
    print(f"\nDecoding {len(imec_tokens)} tokens...")
    encrypted_bits = imec.decode_imec(
        imec_tokens,
        context,
        n_blocks,
        block_size_bits=12
    )
    
    print(f"✓ iMEC decoding complete:")
    print(f"  Output: {len(encrypted_bits)} bits")
    print(f"  Expected: {total_bits} bits")
    
    # ========================================================================
    # STEP 2: DECRYPTION
    # ========================================================================
    
    print("\n" + "="*80)
    print("STEP 2: DECRYPTION (ONE-TIME PAD)")
    print("="*80)
    
    # XOR decryption - handle length mismatch
    decrypt_length = min(len(encrypted_bits), len(one_time_pad_key))
    
    decrypted_bits = ''.join(
        str(int(encrypted_bits[i]) ^ int(one_time_pad_key[i]))
        for i in range(decrypt_length)
    )
    
    print(f"✓ Decryption complete:")
    print(f"  Encrypted: {len(encrypted_bits)} bits")
    print(f"  Decrypted: {len(decrypted_bits)} bits")
    
    if len(decrypted_bits) < total_bits:
        print(f"  ⚠️  Warning: Decrypted fewer bits than expected")
    
    # ========================================================================
    # STEP 3: SIGNAL RECONSTRUCTION
    # ========================================================================
    
    print("\n" + "="*80)
    print("STEP 3: SIGNAL RECONSTRUCTION (BITS → SIGNAL)")
    print("="*80)
    
    recovered_signal = reconstruct_from_bits(decrypted_bits, quant_metadata)
    
    print(f"✓ Signal reconstructed:")
    print(f"  Samples: {len(recovered_signal)}")
    print(f"  Expected: {quant_metadata['num_samples']} samples")
    print(f"  Range: [{recovered_signal.min():.3f}, {recovered_signal.max():.3f}]")
    print(f"  Mean: {recovered_signal.mean():.3f}")
    print(f"  Std: {recovered_signal.std():.3f}")
    
    # Compare with original signal (if available)
    if 'combined_signal' in encoded_data:
        original_signal = encoded_data['combined_signal']
        
        # Truncate to same length
        min_len = min(len(original_signal), len(recovered_signal))
        orig_trunc = original_signal[:min_len]
        recv_trunc = recovered_signal[:min_len]
        
        mse = np.mean((orig_trunc - recv_trunc) ** 2)
        print(f"\nReconstruction quality:")
        print(f"  MSE: {mse:.6f}")
        print(f"  RMSE: {np.sqrt(mse):.6f}")
    
    # ========================================================================
    # STEP 4: FFT ANALYSIS & MESSAGE EXTRACTION
    # ========================================================================
    
    print("\n" + "="*80)
    print("STEP 4: MESSAGE EXTRACTION (FFT + BANDPASS + ASK DECODE)")
    print("="*80)
    
    # Plot FFT
    plot_fft_analysis(recovered_signal, sample_rate, 
                     "FFT Analysis of Recovered Signal")
    
    # Extract each agent's message
    recovered_messages = {}
    accuracies = {}
    
    for agent, freq in agent_frequencies.items():
        print(f"\n{agent} (carrier: {freq} Hz):")
        
        # Define bandpass filter range
        bandwidth = 0.4  # Hz
        freq_low = freq - bandwidth
        freq_high = freq + bandwidth
        
        print(f"  Bandpass filter: [{freq_low:.2f}, {freq_high:.2f}] Hz")
        
        # Apply bandpass filter
        filtered = bandpass_filter(recovered_signal, freq_low, freq_high, sample_rate)
        
        # Decode ASK
        original_bits = messages[agent]
        recovered_bits = decode_ask(filtered, len(original_bits), sample_rate)
        
        # Calculate accuracy
        accuracy = calculate_accuracy(original_bits, recovered_bits)
        
        recovered_messages[agent] = recovered_bits
        accuracies[agent] = accuracy
        
        # Display results
        print(f"  Original:  {original_bits}")
        print(f"  Recovered: {recovered_bits}")
        print(f"  Accuracy:  {accuracy*100:.1f}% ({int(accuracy*len(original_bits))}/{len(original_bits)} bits correct)")
        
        if accuracy >= 0.9:
            print(f"  ✓ Excellent recovery!")
        elif accuracy >= 0.75:
            print(f"  ⚠️  Good recovery (some errors)")
        else:
            print(f"  ❌ Poor recovery")
    
    # ========================================================================
    # SUMMARY
    # ========================================================================
    
    print("\n" + "="*80)
    print("✓ DECODING COMPLETE!")
    print("="*80)
    
    print(f"\nPipeline summary:")
    print(f"  1. iMEC decode:    {len(imec_tokens)} tokens → {len(encrypted_bits)} bits")
    print(f"  2. Decrypt:        {len(decrypted_bits)} bits recovered")
    print(f"  3. Reconstruct:    {len(recovered_signal)} samples")
    print(f"  4. FFT extract:    3 agents recovered")
    
    print(f"\nMessage recovery results:")
    overall_accuracy = np.mean(list(accuracies.values()))
    
    for agent in ['ALICE', 'BOB', 'CHARLIE']:
        acc = accuracies[agent]
        status = "✓" if acc >= 0.9 else "⚠️" if acc >= 0.75 else "❌"
        print(f"  {status} {agent:8s}: {acc*100:.1f}%")
    
    print(f"\n  Overall: {overall_accuracy*100:.1f}%")
    
    if overall_accuracy >= 0.9:
        print("\n✓ SUCCESS: All messages recovered with high accuracy!")
    elif overall_accuracy >= 0.75:
        print("\n⚠️  PARTIAL SUCCESS: Most messages recovered")
    else:
        print("\n❌ FAILURE: Poor message recovery")
    
    print("\n" + "="*80)


if __name__ == "__main__":
    main()
