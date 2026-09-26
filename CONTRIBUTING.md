# Contributing to EDUS

## Workflow

1. Before adding something new, check that it is not already in the code.
2. Never work directly on `main`: create a new branch from `main`, with a meaningful name, so that we remember what it was for even years later (e.g. `feat/phonons`, `fix/velocity-units`).
3. Keep each branch small and focused on one change.
4. Before pushing, make sure you did not break the existing code: build everything and run `ctest`. If a test fails, check what you changed in the parts of the code that were working before (unless you found a bug, in which case update the reference outputs and explain why).
5. Commit only the files you really want to change:
   - `git status` shows what changed;
   - `git diff <file>` shows the changes in a file;
   - `git restore <file>` discards unwanted changes (even just spaces).
6. When you are done, open a pull request towards `main`.

## Code structure

The simulation is split into components, each one printing its own section of the input recap:

- `System` (`src/Electrons`): tight-binding model, band structure, scissor operator;
- `MeanField` (`src/MeanField`): Hartree and screened-exchange self energy, Coulomb interaction;
- `Propagator` (`src/Propagator`): time propagation of the density matrix;
- `OutputManager` (`src/Output`): observables and output files;
- `Simulation` (`src/Simulation`): puts everything together.

### Adding a class

A new class needs two files:

1. A header `.hpp` with the declaration of the class and of all its methods (return type, name and arguments, without the definition). It is included wherever the class is needed with `#include "file.hpp"`; protect it with an include guard (`#ifndef ... #define ... #endif`) so that it is compiled only once.
2. A source `.cpp` with the definitions of everything declared in the header. Add it to the list `_SOURCES` in `CMakeLists.txt` so that it is compiled with the code.

### Adding an input variable

1. Add the variable to `src/InputVariables/input_schema.json`, with its `type` (e.g. `"string"`, `"number"`, `"array"`), a `default` value (used when the variable is not in the input) and a description in `title`.
2. Add the getter and setter to `src/InputVariables/config.hpp`:

   ```cpp
   /// <title>
   inline auto <parameter-name>() const
   {
       return dict_.at("/<parameter-name>"_json_pointer).get<<parameter-type>>();
   }
   inline void <parameter-name>(<parameter-type> <parameter-name>__)
   {
       if (dict_.contains("locked")) {
           throw std::runtime_error(locked_msg);
       }
       dict_["/<parameter-name>"_json_pointer] = <parameter-name>__;
   }
   ```

3. Use the variable in the component it belongs to (otherwise it is read and ignored), and print it in that component's `print_recap()` to keep track of it.

## Releases

Versions follow [semantic versioning](https://semver.org): `MAJOR.MINOR.PATCH`.

- **PATCH**: bug fixes and refactoring, no new features;
- **MINOR**: new features, old inputs still work;
- **MAJOR**: changes that break old inputs.

For each release, add the notes in `docs/releases/vX.Y.Z.md`, merge into `main`, then create the tag and the GitHub release from that file:

```bash
git tag -a vX.Y.Z -m "vX.Y.Z"
git push origin vX.Y.Z
gh release create vX.Y.Z --verify-tag --title "vX.Y.Z" --notes-file docs/releases/vX.Y.Z.md
```
