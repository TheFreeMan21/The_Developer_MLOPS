package main

import (
	"dagger/pipeline/internal/dagger"
)

type Pipeline struct{}

func (m *Pipeline) Base(
	// +defaultPath="/"
	// +ignore=[".git", "**/__pycache__", "**/.venv", "notebooks/*.ipynb", "docs"]
	source *dagger.Directory,
) *dagger.Container {
	return dag.Container().
		From("python:3.13-slim").
		WithMountedCache("/root/.cache/pip", dag.CacheVolume("pip-cache")).
		WithWorkdir("/app").
		WithExec([]string{"pip", "install", "torch", "torchvision", "--index-url", "https://download.pytorch.org/whl/cpu"}).
		WithFile("/app/requirements.txt", source.File("notebooks/requirements.txt")).
		WithExec([]string{"pip", "install", "-r", "requirements.txt"}).
		WithDirectory("/app", source.Directory("py_files/training"))
}
