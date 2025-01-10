


## Dependências

- [Python](https://www.python.org/]) versão 3.10.x
- [Git](https://git-scm.com/)
- [Pygame](https://www.pygame.org/news)
- [Gymnasium](https://gymnasium.farama.org/index.html) versão 0.29.1
- [Protobuf](https://protobuf.dev/) versão 3.20
- [rSoccer](https://github.com/goncamateus/rSoccer)
- [PyVirtualDisplay](https://github.com/ponty/PyVirtualDisplay) versão 3.0 ou acima
- [MoviePy](https://pypi.org/project/moviepy/) versão 1.0.0 ou acima
- [Numpy](https://numpy.org/) versão 1.21.2
- [Argparse](https://docs.python.org/3.10/library/argparse.html)

Exceto o Python e o Git, as dependências podem ser instaladas com:

```bash
  pip install -r requirements.txt
```


```

### Windows (WSL)

Será necessário usar o WSL (Windows Subsystem for Linux) para ser capaz de rodar o projeto no Windows.

1. Instale o WSL. 
[Esse tutorial](https://medium.com/@charles.guinand/installing-wsl2-python-and-virtual-environments-on-windows-11-with-vs-code-a-comprehensive-guide-32db3c1a5847#:~:text=4.2%20Install%20the%20WSL%20Extension,%E2%80%9D%20and%20click%20%E2%80%9CInstall.%E2%80%9D) explica como instalar o WSL, o Python e como fazer a integração com o Visual Studio Code.

2. Crie um [fork](https://docs.github.com/pt/pull-requests/collaborating-with-pull-requests/working-with-forks/fork-a-repo) desse repositório.

3. Clone o repositório.
```bash
  git clone https://github.com/NomeDoUsuario/software-project.git
```

4. Entre na diretório do repositório clonado.
```bash
  cd software-project
```

5. Dentro da pasta, use o comando de instalação das dependências.
```bash
  pip install -r requirements.txt
```



## Run

```bash
  python3 train.py 
```

```bash
  python3 eval.py 
```



