resource "container" "python" {
  disabled = true

  network {
    id = resource.network.local.meta.id
  }

  image {
    name = "nicholasjackson/python-llm:v0.0.1"
  }

  command = ["tail","-f","/dev/null"]
  entrypoint = []

  # add the training file
  volume {
    source      = "./working/train.py"
    destination = "/usr/src/train.py"
  }

  # mount the models directory 
  volume {
    source      = "./working/models"
    destination = "/usr/src/models"
  }

  # mount the data directory 
  volume {
    source      = "./working/data"
    destination = "/usr/src/data"
  }

  # mount the output directory 
  volume {
    source      = "./working/output"
    destination = "/usr/src/output"
  }

  volume {
    source      = "./cache/huggingface"
    destination = "/root/.cache/huggingface"
  }
  
  volume {
    source      = "./cache/ollama"
    destination = "/root/.cache/ollama"
  }

  resources {
    gpu {
      driver     = "nvidia"
      device_ids = ["0"]
    }
  }

  environment = {
    VAULT_ADDR = "http://${resource.ingress.vault_http.local_address}"
    VAULT_TOKEN = resource.exec.vault_init.output.vault_token
    DOCKER_USERNAME = env("DOCKER_USERNAME")
    DOCKER_PASSWORD = env("DOCKER_PASSWORD")
  }
}

#kapsule build \
#  -t docker.io/nicholasjackson/mistral:trained-enc \
#  --encryption-vault-addr ${VAULT_ADDR} \
#  --encryption-vault-auth-token ${VAULT_TOKEN} \
#  --encryption-vault-key kapsule \
#  --encryption-vault-path transit \
#  --username ${DOCKER_USERNAME} \
#  --password ${DOCKER_PASSWORD} \
#  -f ./models/modelfile \
#  ./output/final

//kapsule pull \
// --encryption-vault-addr ${VAULT_ADDR} \
// --encryption-vault-auth-token ${VAULT_TOKEN} \
// --encryption-vault-key kapsule \
// --encryption-vault-path transit \
// --username ${DOCKER_USERNAME} \
// --password ${DOCKER_PASSWORD} \
// --format ollama \
// --output /root/.cache/ollama/models \
// docker.io/nicholasjackson/mistral:trained-enc