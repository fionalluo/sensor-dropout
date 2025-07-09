import torch

def mask_observations_for_student(obs, student_keys, teacher_keys, device=None, debug=False):
    """
    Mask observations for the student policy.
    For each key in student_keys:
      - If the key is in teacher_keys, use obs[key]
      - Else, if key_unprivileged is in teacher_keys, use obs[key_unprivileged]
      - Else, zero out (with correct shape if possible)
    Args:
        obs: dict of all available observations (from environment)
        student_keys: list of keys the student expects
        teacher_keys: list of keys the teacher/expert provides (i.e., what is in obs)
        device: torch device (optional)
        debug: whether to print debug information
    Returns:
        masked_obs: dict with only the student_keys, values as tensors
    """
    masked_obs = {}
    if debug:
        print(f"[MASKING DEBUG] Student keys: {student_keys}")
        print(f"[MASKING DEBUG] Teacher keys: {teacher_keys}")
        print(f"[MASKING DEBUG] Available obs keys: {list(obs.keys())}")
    
    for key in student_keys:
        if key in teacher_keys:
            v = obs[key]
            if debug:
                print(f"[MASKING DEBUG] Key '{key}': direct match, shape {v.shape if hasattr(v, 'shape') else 'scalar'}")
        else:
            # Look for any key in teacher_keys that starts with 'key_unprivileged'
            unprivileged_key = None
            for teacher_key in teacher_keys:
                if teacher_key.startswith(f'{key}_unprivileged'):
                    unprivileged_key = teacher_key
                    break
            
            if unprivileged_key:
                v = obs[unprivileged_key]
                if debug:
                    print(f"[MASKING DEBUG] Key '{key}': using '{unprivileged_key}', shape {v.shape if hasattr(v, 'shape') else 'scalar'}")
            else:
                # Zero out the key - use the shape of the key from the original obs
                if key in obs:
                    shape = obs[key].shape
                else:
                    # If the key doesn't exist in obs at all, use a default shape
                    shape = (1,)
                v = torch.zeros(shape, device=device, dtype=torch.float32)
                if debug:
                    print(f"[MASKING DEBUG] Key '{key}': zeroing out, shape {shape}")
        
        # Ensure tensor and float32
        if not isinstance(v, torch.Tensor):
            v = torch.tensor(v, device=device, dtype=torch.float32)
        elif v.dtype != torch.float32:
            v = v.float()
        
        masked_obs[key] = v
    
    return masked_obs 