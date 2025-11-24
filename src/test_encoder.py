import torch
import torch.nn as nn
# Assuming your encoder class is in a file named 'encoder.py'
from agent.encoder import IdentityEncoder

def verify_weight_tying():
    print("Initializing encoders...")
    # Setup dimensions
    obs_shape = (20,) # Dummy vector observation size
    feature_dim = 128
    num_layers = 4
    
    # 1. Create two distinct encoders
    source_encoder = IdentityEncoder(obs_shape, feature_dim, num_layers, num_filters=32)
    target_encoder = IdentityEncoder(obs_shape, feature_dim, num_layers, num_filters=32)

    # 2. Check Pre-Conditions (Should be different)
    # We check the first layer's weights
    w_src = source_encoder.layers[0].weight
    w_trg = target_encoder.layers[0].weight
    
    print(f"\n[Pre-Tie] Weights are the same object? {w_src is w_trg}")
    print(f"[Pre-Tie] Weights have equal values?   {torch.equal(w_src, w_trg)}")
    
    # 3. Run the Function
    print("\n>>> Calling copy_conv_weights_from(source_encoder)...")
    target_encoder.copy_conv_weights_from(source_encoder)

    # 4. Check Post-Conditions (Should be identical)
    all_passed = True
    
    print("\n[Post-Tie] Checking all layers...")
    for i, (layer_src, layer_trg) in enumerate(zip(source_encoder.layers, target_encoder.layers)):
        # Check Weight Object Identity
        if layer_src.weight is not layer_trg.weight:
            print(f"  X Layer {i} weights are NOT tied by reference!")
            all_passed = False
        else:
            print(f"  ✓ Layer {i} weights are tied.")
            
        # Check Bias Object Identity
        if layer_src.bias is not layer_trg.bias:
            print(f"  X Layer {i} biases are NOT tied by reference!")
            all_passed = False

    # 5. Functional Test (The "Magic" Check)
    # If we modify the source manually, the target should change instantly 
    # because they point to the same memory.
    print("\n[Functional Test] Modifying source encoder weights...")
    with torch.no_grad():
        source_encoder.layers[0].weight.add_(100.0) # Add 100 to all weights in source
    
    # Check if target "saw" the change
    diff = torch.abs(source_encoder.layers[0].weight - target_encoder.layers[0].weight).sum()
    
    if diff == 0:
        print("  ✓ Target changed with Source. Difference is 0.0.")
    else:
        print(f"  X Target did not change! Difference is {diff.item()}.")
        all_passed = False

    if all_passed:
        print("\nSUCCESS: Encoders are perfectly tied.")
    else:
        print("\nFAILURE: Something went wrong.")

if __name__ == "__main__":
    verify_weight_tying()