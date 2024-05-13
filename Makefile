REPO=nicholasjackson/python-llm
VERSION=v0.0.1

build_and_push_python:
	docker run --rm --privileged multiarch/qemu-user-static --reset -p yes
	docker buildx create --name vscode || true
	docker buildx use vscode
	docker buildx inspect --bootstrap
	docker buildx build --platform linux/arm64,linux/amd64 \
		-f ./Dockerfiles/python/Dockerfile \
		-t ${REPO}:${VERSION} \
		--build-arg="HUGGINGFACE_TOKEN=${HUGGINGFACE_TOKEN}" \
	  ./Dockerfiles/python \
		--push
	docker buildx rm -f vscode

build_local_python:
	docker run --rm --privileged multiarch/qemu-user-static --reset -p yes
	docker buildx create --name vscode || true
	docker buildx use vscode
	docker buildx inspect --bootstrap
	docker buildx build \
		-f ./Dockerfiles/python/Dockerfile \
		-t ${REPO}:${VERSION} \
		--build-arg="HUGGINGFACE_TOKEN=${HUGGINGFACE_TOKEN}" \
	  ./Dockerfiles/python \
		--load
	docker buildx rm -f vscode

build_mistral_with_kapsule_ollama:
	kapsule build \
		--debug \
		--output ./cache/ollama/models \
		--format ollama \
		-f ./working/models/base/Modelfile \
		-t auth.demo.gs:5001/nicholasjackson/mistral:plain \
		./working/output/final

build_mistral_with_kapsule_oci:
	kapsule build \
		--debug \
		-f ./working/models/base/Modelfile \
		-t auth.container.local.jmpd.in:5001/nicholasjackson/mistral:plain \
		--username=admin \
		--password=password \
		--insecure \
		./working/output/final

build_trained_with_kapsule_ollama:
	kapsule build \
		--debug \
		--output ./cache/ollama/models \
		--format ollama \
		-f ./working/models/trained/Modelfile \
		-t auth.demo.gs:5001/nicholasjackson/trained:plain \
		./working/output/final

build_trained_with_kapsule_oci:
	kapsule build \
		--debug \
		-f ./working/models/trained/Modelfile \
		-t auth.demo.gs:5001/nicholasjackson/trained:plain \
		--username=admin \
		--password=password \
		--insecure \
		./working/output/final


pull_mistral_with_kapsule_ollama:
	kapsule pull \
		--debug \
		--output ./cache/ollama/models \
		--format ollama \
		--username=admin \
		--password=password \
		--insecure \
		auth.container.local.jmpd.in:5001/nicholasjackson/mistral:plain \