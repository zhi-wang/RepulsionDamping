# Comparing different implementations of the repulsion damping coefficients in the HIPPO force field

## Compile, Run, and Clean

```bash
make              # compile
./a.out input.txt # run
make clean        # clean
```

## Lines in the input file
```
# The input file can hold multiple lines of data.
# either
dampi  dampj     xi     yi     zi     xj     yj     zj
# or
dampi  dampj    rij
```
