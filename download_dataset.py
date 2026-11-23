
from huggingface_hub import snapshot_download # importa a função de download de um repositório inteiro do hugging face

# Usando a função para importar dataset (para mais informações, consultar a documentacao do hugging face)
caminho_download = snapshot_download(repo_id="allenai/WildChat-1M", repo_type="dataset",
                            local_dir="./WildChat",
                            allow_patterns="data/*.parquet", dry_run=False) #repo id, que é o nome do dataset, e tipo de repositório, que é dataset, e, por fim, define o local de download

print(f"Donwlods realizados em {caminho_download}")
