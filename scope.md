# Scope and Intent: Science Genome

## Purpose of the Project
The Science Genome project exists to design and propose an effective method for studying how scientific ideas evolve. It is a research effort focused on building a scientifically rigorous framework that researchers can use to run investigations and generate defensible insights.

The central objective is not software production; it is methodological development. The project should produce a clear way to represent, trace, and analyze the evolution of scientific ideas across time, while preserving interpretability and empirical discipline.

## Relationship Between Paper and Code
The paper and the code serve distinct but complementary roles.

- **The paper is the primary source of truth.**
- **The code is an executable companion to the paper.**

The paper provides the conceptual and mathematical formulation. The code realizes that formulation so it can be explored, tested, inspected, and used to generate supporting evidence.

The code does **not** replace the mathematics, and the paper’s conceptual legitimacy does **not** depend on the software. Instead, the implementation supports the research process by:

- checking whether the proposed mathematics coheres in practice,
- generating figures, statistics, and diagnostics,
- enabling exploratory investigations of idea evolution,
- helping refine the framework through empirical feedback.

In short, the code should make the framework tangible without becoming the framework itself.

## Purpose of the Paper
The paper should clearly and concisely present the proposed method and provide evidence for its legitimacy and robustness in producing meaningful insights.

Its communication priorities, in decreasing order, are to:

1. reassure the reader of correctness, robustness, and confidence in accurate formulation,
2. showcase the framework and its analytical structure,
3. excite the reader about what the method makes possible,
4. excite the reader about the method’s broader potential.

The introduction and methodology sections define the conceptual center of gravity for this project and should guide scope decisions.

## Purpose of the Code
The codebase exists to support the paper by implementing its conceptual and mathematical framework as research-grade executable artifacts.

Its value is in enabling verification, demonstration, and refinement of the proposed method—not in becoming an independent software product.

## Design Philosophy
The code should be **clear, concise, and correct**. Every implementation choice should directly support research intent.

No line of code should exist only for speculative flexibility, architectural neatness, or future features that are not currently required.

Correctness means:

- faithfulness to the paper,
- transparent inspectability,
- traceability from concepts and equations to implementation and outputs.

Where possible, names and abstractions in code should mirror the conceptual objects in the paper.

Preferred trade-offs:

```text
Clarity over speed.
Minimalism over extensibility.
Ease of use over software completeness.
Exploratory flexibility over rigid formalisation.
Conceptual abstraction over procedural sprawl.
Paper-alignment over independent software design.
```

Abstraction should be used only when it improves conceptual clarity or removes repeated mathematical logic. The code should be conceptually abstract, not architecturally abstract.

Dependencies should be minimal. Logic should be simple. Modules should be small. The code should avoid hidden machinery, unnecessary robustness layers, and feature creep.

## Scope and Non-Goals
This is research code for developing, testing, and demonstrating the proposed framework.

- **Primary user:** the author during investigation and paper development.
- **Secondary users:** researchers who may inspect, adapt, or reuse parts of the implementation.

### In Scope

```text
Implementing the mathematical framework.
Testing whether the mathematics works coherently in practice.
Producing figures, statistics, and diagnostics for the paper.
Supporting exploratory investigation.
Making the framework inspectable and reusable in parts.
Keeping the implementation aligned with the paper.
```

### Out of Scope

```text
Production-grade software.
A general-purpose scientometrics toolkit.
A web application.
A graphical user interface.
Large-scale automation.
Enterprise robustness.
Plugin systems or broad extensibility layers.
Premature optimisation.
Features not required by the research.
```

## Scope Test for New Work
A proposed change is in scope only if it directly improves at least one of:

- fidelity of implementation to the paper,
- rigor or clarity of verification/validation,
- quality of figures, statistics, or diagnostics that support the paper,
- usefulness for exploratory scientific-idea evolution analysis,
- inspectability and reproducibility of research outputs.

If it does not, it should be deferred or treated as a separate effort.

## Success Condition
The project succeeds when it remains a lean executable realization of the paper’s ideas and helps answer the research question with credible, inspectable evidence.

It should strengthen confidence in the framework, support method refinement, and produce artifacts that make the proposed approach scientifically understandable and practically investigable.
