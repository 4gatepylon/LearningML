I read through this book (edition 5): https://www.amazon.com/Programming-Massively-Parallel-Processors-Hands/dp/0323912311

I did the exercises as I went through and wrote them down below in this folder.


## Running things

Each `.cc` file is a standalone executable with its own `main`. Build and run one with:

```
make run FILE=<name>    # <name> is the filename without the .cc extension
```

For example, `make run FILE=vector_add`. Run `make help` to see all targets.

Object files and binaries are stored in `./.artifacts/` (gitignored). `make clean` removes that folder.
