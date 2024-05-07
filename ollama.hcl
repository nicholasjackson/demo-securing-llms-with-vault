resource "container" "ollama" {
  image {
    name = "ollama/ollama:latest"
  }

  network {
    id = resource.network.local.meta.id
  }

  resources {
    gpu {
      driver     = "nvidia"
      device_ids = ["0"]
    }
  }

  volume {
    //source      = data("ollama")
    source      = "./cache/ollama"
    destination = "/root/.ollama"
  }
  
  volume {
    source      = "./working"
    destination = "/working"
  }

  port {
    host  = 11434
    local = 11434
  }
}

resource "container" "ollama_ui" {
  network {
    id = resource.network.local.meta.id
  }

  image {
    name = "ghcr.io/open-webui/open-webui:main"
  }

  volume {
    source      = "./cache/open-webui"
    destination = "/app/backend/data"
  }

  environment = {
    OLLAMA_BASE_URL = "http://${resource.container.ollama.container_name}:11434"
  }

  port {
    host  = 8080
    local = 8080
  }
}