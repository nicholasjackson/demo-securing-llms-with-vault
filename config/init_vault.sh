set -e

# Wait for vault to be ready, we determine ready as the server reporting sealed = true
until [ "$(vault status --format json | jq .sealed)" = "true" ]; do
  echo "Vault is not running, waiting 2 seconds..."
  sleep 2
done

# Initialize vault and store the unseal key and root token
vault operator init -key-shares=1 -key-threshold=1 -format=json > /tmp/vault_init.json
echo "vault_unseal_key=$(cat /tmp/vault_init.json | jq -r .unseal_keys_b64[0])" >> ${EXEC_OUTPUT}
echo "vault_token=$(cat /tmp/vault_init.json | jq -r .root_token)" >> ${EXEC_OUTPUT}

# Unseal vault
vault operator unseal $(cat /tmp/vault_init.json | jq -r .unseal_keys_b64[0])

export VAULT_TOKEN=$(cat /tmp/vault_init.json | jq -r .root_token)

# Enable transit secrets engine
vault secrets enable transit

# Create a new encryption key
vault write transit/keys/kapsule exportable=true type=rsa-4096

