#!/usr/bin/env python3
import numpy as np
import sys

tf_out = np.fromfile('xla_hlo/out/sd_unet_tiny_tf_output.raw.f32', dtype=np.float32)
sp_out = np.fromfile('xla_hlo/out/sd_unet_spirv_local.raw.f32', dtype=np.float32)
print('TF output (first 16):', tf_out[:16])
print('SPIRV output (first 16):', sp_out[:16])
print('Diff (first 16):', np.abs(tf_out - sp_out)[:16])
print('TF range:', tf_out.min(), tf_out.max(), tf_out.mean())
print('SPIRV range:', sp_out.min(), sp_out.max(), sp_out.mean())
print('Max abs err:', np.max(np.abs(tf_out - sp_out)))
print('Elements >0.001:', np.sum(np.abs(tf_out - sp_out) > 0.001), '/', len(tf_out))
print('Elements >0.01:', np.sum(np.abs(tf_out - sp_out) > 0.01), '/', len(tf_out))
print('Elements >0.1:', np.sum(np.abs(tf_out - sp_out) > 0.1), '/', len(tf_out))
