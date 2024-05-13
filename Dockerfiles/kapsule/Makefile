REPO=nicholasjackson/kapsule
VERSION=v0.0.7

build_and_push:
	docker run --rm --privileged multiarch/qemu-user-static --reset -p yes
	docker buildx create --name vscode || true
	docker buildx use vscode
	docker buildx inspect --bootstrap
	docker buildx build --platform linux/arm64,linux/amd64 \
		-t ${REPO}:${VERSION} \
	  . \
		--push

build_local:
	docker build -t ${REPO}:${VERSION} .