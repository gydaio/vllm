import os
import boto3
from vllm.lora.request import LoRARequest
from vllm.lora.resolver import LoRAResolver, LoRAResolverRegistry


class S3LoRAResolver(LoRAResolver):
    def __init__(self):
        self.s3 = boto3.client('s3')
        self.lora_cache_dir = os.getenv("VLLM_LORA_RESOLVER_CACHE_DIR")

    async def resolve_lora(self, base_model_name, lora_name):
        local_path = os.path.join(self.lora_cache_dir, lora_name)

        try:
            # Download the LoRA from S3 to the local path
            bucket = os.getenv("S3_LORA_BUCKET")
            base_model_name = os.getenv("MODEL").replace("/", "-")
            lora_path = os.path.join(base_model_name, lora_name)

            # Make local folder
            os.makedirs(os.path.dirname(local_path), exist_ok=True)

            # Download LoRA
            self.s3.download_file(bucket, lora_path, local_path)

            lora_request = LoRARequest(
                lora_name=lora_name,
                lora_path=local_path,
                lora_int_id=abs(hash(lora_name))
            )
            return lora_request

        except Exception as e:
            print(f"Error downloading LoRA from S3: {e}")
            return None


def register_s3_resolver():
    s3_resolver = S3LoRAResolver()
    LoRAResolverRegistry.register_resolver("s3_resolver", s3_resolver)
