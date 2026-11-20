package main

import (
	"dagger/pipeline/internal/dagger"
)

type Pipeline struct {
	// +private
	Source *dagger.Directory
}

func New(
	// +defaultPath="/"
	// +ignore=[".git", "**/__pycache__", "**/.venv", "notebooks/*.ipynb", "docs"]
	source *dagger.Directory,
) *Pipeline {
	return &Pipeline{Source: source}
}

func (m *Pipeline) Base() *dagger.Container {
	return dag.Container().
		From("python:3.13-slim").
		WithMountedCache("/root/.cache/pip", dag.CacheVolume("pip-cache")).
		WithWorkdir("/app").
		WithExec([]string{"pip", "install", "torch", "torchvision", "--index-url", "https://download.pytorch.org/whl/cpu"}).
		WithFile("/app/requirements.txt", m.Source.File("notebooks/requirements.txt")).
		WithExec([]string{"pip", "install", "-r", "requirements.txt"}).
		WithDirectory("/app", m.Source.Directory("py_files/training"))
}

func (m *Pipeline) Data() *dagger.Directory {
	return m.Base().
		WithDirectory("/app/.dvc", m.Source.Directory(".dvc")).
		WithFile("/app/.dvcignore", m.Source.File(".dvcignore")).
		WithDirectory("/app/data", m.Source.Directory("data")).
		WithExec([]string{"dvc", "config", "--local", "core.no_scm", "true"}).
		WithExec([]string{"dvc", "checkout"}).
		Directory("/app/data")
}
