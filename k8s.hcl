resource "k8s_cluster" "k3s" {
  network {
    id = resource.network.local.meta.id
  }
}

resource "helm" "vault" {
  cluster = resource.k8s_cluster.k3s

  repository {
    name = "hashicorp"
    url  = "https://helm.releases.hashicorp.com"
  }

  chart   = "hashicorp/vault" # When repository specified this is the name of the chart
  version = "v0.27.0"         # Version of the chart when repository specified

  values = "./config/vault-values.yaml"
}

resource "ingress" "vault_http" {
  port = 8200

  target {
    resource = resource.k8s_cluster.k3s
    port     = 8200

    config = {
      service   = "vault"
      namespace = "default"
    }
  }
}

# Configure the Vault Kubernetes service account
resource "k8s_config" "vault_auth" {
  cluster = resource.k8s_cluster.k3s

  docker {
    no_proxy  = ["${resource.container.auth.container_name}:5001"]
  }

  paths = [
    "./config/vault_k8s_service_account.yaml",
  ]

  wait_until_ready = true
}

# Initialize the Vault server and configure
resource "exec" "vault_init" {
  depends_on = [ "resource.helm.vault" ]

  image {
    name = "shipyardrun/hashicorp-tools:v0.11.0"
  }

  script = file("./config/init_vault.sh")

  volume {
    source = resource.k8s_cluster.k3s.kube_config.path
    destination = "/root/.kube/config"
  }

  environment = {
    VAULT_ADDR = "http://${resource.ingress.vault_http.local_address}"
    KUBECONFIG = "/root/.kube/config"
  }
}

//resource "container" "vault_init" {
//  depends_on = ["resource.helm.vault"]
//
//  image {
//    name = "shipyardrun/hashicorp-tools:v0.11.0"
//  }
//
//  command = [
//    "tail",
//    "-f",
//    "/dev/null",
//  ]
//
//  volume {
//    source      = resource.k8s_cluster.k3s.kube_config.path
//    destination = "/root/.kube/config"
//  }
//
//  environment = {
//    VAULT_ADDR = "http://${resource.ingress.vault_http.local_address}"
//    KUBECONFIG = "/root/.kube/config"
//  }
//}
//
resource "k8s_config" "ollama" {
  depends_on = [ "resource.exec.vault_init" ]

  cluster = resource.k8s_cluster.k3s

  paths = [
    "./files/ollama.yaml",
  ]

  wait_until_ready = true
}

resource "ingress" "ollama_ui" {
  port = 8080

  target {
    resource = resource.k8s_cluster.k3s
    port = 80

    config = {
      service   = "ollama-ui"
      namespace = "default"
    }
  }
}

output "VAULT_ADDR" {
  value = "http://${resource.ingress.vault_http.local_address}"
}

output "VAULT_TOKEN" {
  value = resource.exec.vault_init.output.vault_token
}

output "KUBECONFIG" {
  value = resource.k8s_cluster.k3s.kube_config.path
}
