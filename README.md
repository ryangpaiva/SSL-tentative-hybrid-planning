
# Bem-vindos ao repositório do projeto de software! 🥳

Este repositório concentra toda a base de código que será usado nesta seletiva. Nele, estão disponibilizados tanto o ambiente simulado para testar o código desenvolvido quanto o arquivo base para o desenvolvimento, no qual deverão ser integradas as mudanças que forem feitas. O projeto consiste em desenvolver formas de solucionar os problemas de planejamento de trajetória e de atribuição de tarefas para os agentes do ambiente. Para mais detalhes do projeto, reveja o documento de especificação no nosso site: https://www.robocin.com.br/seletiva.

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

Caso a sua versão do Python não seja a correta, [esse tutorial](https://gist.github.com/rutcreate/c0041e842f858ceb455b748809763ddb) explica como instalar a versão correta no Linux.

## Instalação

### Linux

1. Crie um [fork](https://docs.github.com/pt/pull-requests/collaborating-with-pull-requests/working-with-forks/fork-a-repo) desse repositório.

2. Clone o repositório.
```bash
  git clone https://github.com/NomeDoUsuario/software-project.git
```

3. Entre na diretório do repositório clonado.
```bash
  cd software-project
```

4. Dentro da pasta, use o comando de instalação das dependências.
```bash
  pip install -r requirements.txt
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

### MacOS

1. Instale o [Homebrew](https://brew.sh/) e a versão do [Pyenv](https://github.com/pyenv/pyenv) para instalação do Python

```bash
  brew install pyenv
  pyenv install 3.10
  pyenv global 3.10
```

2. Crie um [fork](https://docs.github.com/pt/pull-requests/collaborating-with-pull-requests/working-with-forks/fork-a-repo) desse repositório.

3. Clone o repositório.
```bash
  git clone https://github.com/NomeDoUsuario/software-project.git
```

4. Entre na diretório do repositório clonado.
```bash
  cd software-project
```

5. Instale a [ODE](https://www.ode.org/) (Open Dynamics Engine)
```bash
  brew install ode
```

6. Dentro da pasta, use o comando de instalação das dependências.
```bash
  pip install -r requirements.txt
```
⚠️ *OBS:* Para rodar no macOS com a configuração acima, utilize o comando `python3.10` ao invés de `python3` no passo a passo a seguir.

Tudo pronto para rodar o projeto! 🚀

## Como rodar?

Para rodar, basta executar o arquivo `start.py`.
```bash
  python3 start.py
```

Como o projeto possui 4 fases, é possível escolher qual fase rodar utilizando a flag `-d` com o argumento de dificuldade, que vai de 1 a 4:

```bash
  python3 start.py -d [DIFICULDADE]
```

Para tirar dúvidas, use o comando com a flag `-h`:

```bash
  python3 start.py -h
```

⚠️ *OBS:* Caso a instalação das dependências não tenha sido feita em um ambiente virtual e os comandos para rodar não estejam funcionando, tente usar `python3.10` ao invés de `python3`. 
⚠️ *OBS:* Caso tenha problemas com a instalação das dependências do pacote `rc-robosim`ou de CMAKE no ambiente Linux, experimente atualizar o sistema de pacotes do sistema e reinstalar a biblioteca ODE (Open Dynamics Engine) com os comandos: 
```bash
  sudo apt update
  sudo apt upgrade
  sudo apt install libode-dev
```


