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
