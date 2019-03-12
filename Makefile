BUILD_VERSION=$(shell echo $(or ${CI_COMMIT_SHA}, LOCAL) | cut -c1-8)

DOCKER_IMAGE=avdetect

.PHONY: build
build:
        docker build -t $(DOCKER_IMAGE):$(BUILD_VERSION) . -f ops/compose/Dockerfile
