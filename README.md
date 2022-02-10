# mthe-493-group-a2

> Optimization of Data Allocation with Training Time Constraints in Heterogeneous Edge Learning

Main repo for MTHE 493 Group A2's documentation and source code. An implementation of a distributed edge learning algorithm that optimizes how data is allocated to workers.

## Getting started

Note: The project dependencies target an environment that is a 32-bit ARM architecture (e.g., a raspberrypi) with python 3.9.

To install necessary project dependencies, run

```shell
git clone https://github.com/bryan-hoang/mthe-493-group-a2.git
cd mthe-493-group-a2
make install
```

on each machine to be used to install the python dependencies using `pipenv` and to ensure proper libraries are installed for `pytorch` to work.

**Access the `pipenv` virtual environment using `pipenv shell` or `pipenv run <...>`**. Then to set up the system,

1. Start the notice board:

   To start the notice board, run

   ```sh
   python src/notice_board.py
   ```

1. Start the workers:

   Run

   ```sh
   python src/worker.py
   ```

   on each machine that you would like to use as a worker.

1. Start the client:

   Run

   ```sh
   python src/client.py
   ```

   on the machine you would like to act as the orchestrator.

## Developing

### Built With

- [axon-ecrg](https://github.com/DuncanMays/axon-ECRG)
- [pytorch](https://github.com/pytorch/pytorch)

### Prerequisites

- `python 3.9`

### Setting up Dev

```shell
git clone https://github.com/bryan-hoang/mthe-493-group-a2.git
cd mthe-493-group-a2
make install-dev
```

The `install-dev` recipe installs additional useful development experience packages list in the project's [Pipfile](Pipfile).

### Deploying / Publishing

<!-- TODO(bryan-hoang): Not sure if/how we're planning on improving this part of the orkflow. -->

give instructions on how to build and release a new version
In case there's some step you have to take that publishes this project to a
server, this is the right time to state it.

```shell
packagemanager deploy your-project -s server.com -u username -p password
```

And again you'd need to tell what the previous code actually does.

## Configuration

<!-- TODO(bryan-hoang): To alter test parameters. -->

TBD.

## Tests

Describe and show how to run the tests with code examples.
Explain what these tests test and why.

```shell
Give an example
```

## Style guide

[black](https://github.com/psf/black).
