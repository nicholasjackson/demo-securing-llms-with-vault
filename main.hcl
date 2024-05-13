variable "docs_url" {
  default = "http://localhost:8080"
}

resource "network" "local" {
  subnet = "10.6.0.0/16"
}

variable "auth_ip_address" {
  default = "10.6.0.183"
}

//docker run --rm -it --gpus=all nvcr.io/nvidia/k8s/cuda-sample:nbody nbody -gpu -benchmark
resource "container" "gpu_test" {
  disabled = true
  image {
    name = "nvcr.io/nvidia/k8s/cuda-sample:nbody"
  }

  command = ["nbody", "-gpu", "-benchmark"]

  resources {
    gpu {
      driver     = "nvidia"
      device_ids = ["0"]
    }
  }
}
