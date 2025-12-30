import bitsandbytes as bnb
import transformers
import accelerate
import torch

print(f"✅ Torch version: {torch.__version__}")
print(f"✅ CUDA available: {torch.cuda.is_available()}")
print(f"✅ BitsAndBytes version: {bnb.__version__}")
print(f"✅ Transformers version: {transformers.__version__}")
print(f"✅ Accelerate version: {accelerate.__version__}")

# Comprobación específica del error que tenías
try:
    from transformers.utils.quantization_config import BitsAndBytesConfig
    print("✅ BitsAndBytesConfig importado correctamente.")
except ImportError as e:
    print(f"❌ Error importando config: {e}")