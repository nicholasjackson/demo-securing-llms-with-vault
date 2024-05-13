#! /bin/bash

set -e

vault secrets enable transit

vault write transit/keys/data exportable=true type=rsa-4096
vault write transit/keys/llm exportable=true type=rsa-4096

vault auth enable userpass

vault write auth/userpass/users/ops \
 password=password \
 policies=ops

 vault write auth/userpass/users/datascience \
 password=password \
 policies=datascience

 vault write auth/userpass/users/runtime \
 password=password \
 policies=runtime

vault policy write ops ../policy/ops.hcl
vault policy write datascience ../policy/datascience.hcl
vault policy write runtime ../policy/runtime.hcl