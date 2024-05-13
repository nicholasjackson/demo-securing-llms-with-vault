resource "template" "vscode_settings" {

  source = <<-EOF
  {
      "workbench.colorTheme": "GitHub Dark",
      "editor.fontSize": 16,
      "workbench.iconTheme": "material-icon-theme",
      "terminal.integrated.fontSize": 16
  }
  EOF

  destination = "${data("vscode")}/settings.json"
}

resource "template" "vscode_jumppad" {

  source = <<-EOF
  {
  "tabs": [
    {
      "name": "Docs",
      "uri": "${variable.docs_url}",
      "type": "browser",
      "active": true
    },
    {
      "name": "Terminal",
      "location": "editor",
      "type": "terminal"
    }
  ]
  }
  EOF

  destination = "${data("vscode")}/workspace.json"
}

resource "container" "vscode" {

  network {
    id = resource.network.local.meta.id
  }

  image {
    name = "nicholasjackson/securing-llms-with-vault:v0.1.0"
  }


  volume {
    source      = "./working"
    destination = "/usr/src/working"
  }

  volume {
    source      = "./cache/huggingface"
    destination = "/root/.cache/huggingface"
  }

  //volume {
  //  source      = resource.template.vscode_jumppad.destination
  //  destination = "/workshop/.vscode/workspace.json"
  //}

  volume {
    source      = resource.template.vscode_settings.destination
    destination = "/usr/src/.vscode/settings.json"
  }

  volume {
    source      = "/var/run/docker.sock"
    destination = "/var/run/docker.sock"
  }

  //volume {
  //  source      = resource.k8s_cluster.dev.kube_config.path
  //  destination = "/root/.kube/config"
  //}

  environment = {
    KUBE_CONFIG_PATH  = "/root/.kube/config"
    KUBECONFIG        = "/root/.kube/config"
    DEFAULT_FOLDER    = "/workshop"
    PATH              = "/usr/local/go/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
    LC_ALL            = "C"
    VAULT_ADDR        = "http://${resource.ingress.vault_http.local_address}"
    HUGGINGFACE_TOKEN = env("HUGGINGFACE_TOKEN")
    DOCKER_USERNAME   = env("DOCKER_USERNAME")
    DOCKER_PASSWORD   = env("DOCKER_PASSWORD")
  }

  // vscode
  port {
    local = 8000
    host  = 8000
  }

  // ssh 
  port {
    local = 22
    host  = 2222
  }

  health_check {
    timeout = "100s"

    http {
      address       = "http://localhost:8000/"
      success_codes = [200, 302, 403]
    }
  }

  //resources {
  //  gpu {
  //    driver     = "nvidia"
  //    device_ids = ["0"]
  //  }
  //}
}