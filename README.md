# Comparing different implementations of the repulsion damping coefficients in the HIPPO force field

[Documentation on overleaf.com](https://www.overleaf.com/read/qnhhwbpctrqn)

## Compile, Run, and Clean

```bash
make              # compile
./a.out input.txt # run
make clean        # clean
```

## Lines in the input file
```
# The input file can hold multiple lines of data.
# X = R: Repulsion
# X = G: Gordon 1
# X = D: Dispersion
# either
X   dampi  dampj     xi     yi     zi     xj     yj     zj
# or
X   dampi  dampj    rij
```
