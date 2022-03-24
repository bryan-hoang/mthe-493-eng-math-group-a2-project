# Architecture

This document describes the high-level architecture of the experimental set up. If you want to familiarize yourself with the code base, you are just in the right place!

## Bird's Eye View

Here's a flowchart diagrams of the network:

```mermaid
flowchart LR
    subgraph w["Worker(s)"]
        direction LR
        w1[Worker 1]-...-wn[Worker n]
    end
    w-->|Sign in|Noticeboard
    Client<-->|"Discover Worker(s)"|Noticeboard
    Client<-->|Deploy work|w
```

Here's a sequence diagram of the events that occur during a typical experiment:

```mermaid
sequenceDiagram
    autonumber
    participant C as Client
    participant W as Worker(s)
    participant N as Noticeboard
    W->>N: Sign in for discovery
    C->>N: Request IP addresses of Worker(s)
    N->>C: Respond with IP addresses of Worker(s)
    C->>+W: Deploy benchmarking work
    W->>-C: Return benchmarking results
    C->>+W: Distribute data optimally given constraints
    loop Every global iteration
        C->W: Perform local updates
    end
    W->>-C: Return and aggregate results
```

On the highest level, this project uses [axon-ecrg](https://github.com/DuncanMays/axon-ECRG#readme) to create a proof of concept framework for parallelizing work through distributed computing, which aims to optimize the amount of data sent to each worker.

## Code Map

This section talks briefly about various important directories/files.
