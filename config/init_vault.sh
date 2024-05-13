set -e

# Wait for vault to be ready, we determine ready as the server reporting sealed = true
until [ "$(vault status --format json | jq .sealed)" = "true" ]; do
  echo "Vault is not running, waiting 2 seconds..."
  sleep 2
done

# Initialize vault and store the unseal key and root token in the outputs
vault operator init -key-shares=1 -key-threshold=1 -format=json > /tmp/vault_init.json
echo "vault_unseal_key=$(cat /tmp/vault_init.json | jq -r .unseal_keys_b64[0])" >> ${EXEC_OUTPUT}
echo "vault_token=$(cat /tmp/vault_init.json | jq -r .root_token)" >> ${EXEC_OUTPUT}

# Unseal vault
vault operator unseal $(cat /tmp/vault_init.json | jq -r .unseal_keys_b64[0])

# Export the root token to an environment variable so we can use it in the next steps
export VAULT_TOKEN=$(cat /tmp/vault_init.json | jq -r .root_token)

# Enable user auth
vault auth enable userpass

# Configure users
vault write auth/userpass/users/ops \
  password=password \
  policies=ops

vault write auth/userpass/users/datascience \
  password=password \
  policies=datascience

vault write auth/userpass/users/runtime \
  password=password \
  policies=runtime

# Enable kubernetes auth
vault auth enable kubernetes

# Configure kubernetes auth
kubectl get secret vault-k8s-auth-secret -o json | jq -r '.data."ca.crt"' | base64 -d > /k8s.crt
kubectl get secret vault-k8s-auth-secret -o json | jq -r '.data.token' | base64 -d > /k8s.token

vault write auth/kubernetes/config \
  token_reviewer_jwt="$(cat /k8s.token)" \
  kubernetes_host="https://kubernetes.default.svc:443" \
  kubernetes_ca_cert=@/k8s.crt