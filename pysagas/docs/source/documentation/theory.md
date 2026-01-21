(theory-docs)=

# Approximating Sensitivities Using Low-Order Models
The core idea underpinning *PySAGAS* is to use low-order, 
differentiable fluid models to evaluate the sensitivities 
of surface pressures to geometric parameters.


## High-Level Overview of Approach

```{tip}
Refer to the [nomenclature](nomenclature) for definitions of 
symbols used here.
```

(vertex-param-sens)=
### Vertex-Parameter Sensitivities
To make the link between [cell](cell-definition) data and geometric parameters, 
*PySAGAS* requires the user to provide the [vertex-parameter sensitivities](geom-param-sens), 
$\partial\underline{v}/\partial\underline{\theta}$. In simple cases, such as the 
[inclined ramp example](example-inclined-ramp), 
$\partial\underline{v}/\partial\underline{\theta}$ can be found manually. However, for more 
[complex geometries](example-nose-opt), this is not possible. For this reason, a parametric 
geometry generation tool offering geometric differentiation is recommended. One such tool
is [*hypervehicle*](https://github.com/kieran-mackle/hypervehicle/).


(nominal-surface-properties)=
### Nominal Surface Properties
*PySAGAS* also requires the nominal surface properties of the geometry being considered. Namely, 
the local Mach number, pressure and temperature. This information can come from any source, making 
*PySAGAS* CFD-solver agnostic; whether it be [Oblique Shock Theory](oblique-shock-relations), or 
[Cart3D](nose-cart3d-modelling).


### Pressure Sensitivities
Given the nominal surface properties, *PySAGAS* can calculate the sensitivity of local pressure 
to changes in the normal vector of a [cell](cell-definition). Once these sensitivities are known, 
they can be linked to the geometric design parameters via application of the chain rule.

Let's start with *piston theory*. Surface pressure is given by the following expression, relative 
to the freestream properties, denoted by the $\infty$ subscript.


$$
    P = P_\infty \, \left( 1 + \frac{\gamma - 1}{2} \, \frac{W}{a_\infty} \right)^{\frac{2 \, \gamma}{\gamma - 1}}
$$


In this expression, $W$ is the downwash speed, $a_{\infty}$ is the 
freestream speed of sound, and $\gamma$ is the ratio of specific 
heats. By following the approach presented in Zhan (2009)[^1], keeping
only first-order terms and decomposing the downwash velocity, we 
obtain "local piston theory":

$$ P = P_l + \rho_l \, a_l \, W$$

$$ W = \underline{v}_l \cdot \delta \underline{n} + \underline{V}_b \cdot \underline{n} $$

$$ \delta \underline{n} = \underline{n}_0 - \underline{n} $$

<!-- $$
    P = P_l \, \left( 1 + \frac{\gamma - 1}{2} \, \frac{v_n}{a_l} \right)^{\frac{2 \, \gamma}{\gamma - 1}},
$$ -->


[^1]: [Zhan, W.-W., Ye, Z.-Y., and Zhang, C.-A., “Supersonic Flutter 
Analysis Based on a Local Piston Theory,” AIAA Journal, Vol. 47, 
No. 10, 2009.](https://arc.aiaa.org/doi/10.2514/1.37750)


Here, the subscript $l$ refers to the local surface 
conditions - i.e. the 
[nominal surface properties](nominal-surface-properties)
as described above.
In this model, we are only interested in the first term, 
$\underline{v}_l \cdot \delta \underline{n}$, as this relates 
to geometric deformation. The second term, 
$\underline{V}_b \cdot \underline{n}$, relates to vibration, 
and can therefore be discarded. We therefore reach an
expression for the local pressure, relative to a cell's
normal vector.

$$ P = P_l + \rho_l \, a_l \, \underline{v}_l \cdot \left( \underline{n}_0 - \underline{n} \right)$$

This expression can be differentiated with respect to 
the geometric parameters $\underline{p}$, yielding the 
sensitivity of pressure to parameters $\underline{p}$:

$$ 
\frac{\partial \, P}{\partial \, \underline{\theta}} = 
\rho_l \, a_l \, \underline{v}_l \cdot 
\left(- \frac{\partial \, \underline{n}}{\partial \, \underline{\theta}} \right)
$$

Of course, this expression requires $d\underline{n}/d\underline{p}$. 
Conveniently, this can be evaluated using the 
[vertex parameter sensitivities](vertex-param-sens) and the 
[normal vector vertex sensitivities](normal-area-vertex-sens)
as follows:

$$ 
\frac{\partial \, \underline{n}}{\partial \, \underline{\theta}} = \frac{\partial \, \underline{n}}{\partial \, 
\underline{v}}  \frac{\partial \, \underline{v}}{\partial \, \underline{\theta}} 
$$



### Aerodynamic Sensitivities
With the sensitivity of pressure to parameters, $\partial P/ \partial \underline{\theta}$,
*PySAGAS* is able to calculate the sensitivity of the aerodynamic
surface properties to the geometric design parameters
(i.e. $\frac{\partial \, C_X}{\partial \, \underline{\theta}}$). First, note that the 
force acting on a given cell is the product of the pressure acting 
on that cell, and the cell's area. The total force acting on the 
geometry is therefore the sum of cell forces:

<!-- the sum notation here can be improved for clarity -->


$$
\underline{F}_{\text{ i}} =  P_i(\underline{\theta}) \, A_i(\underline{\theta}) \, 
\underline{n}_{i}(\underline{\theta})
$$

$$
\underline{F}_{\text{ net}} = \sum_{i=0}^{N} 
\underline{F}_{\text{ i}}
$$

This expression can be differentiated with respect to the geometric
parameters to yield the following.

$$
    \frac{\partial\underline{F}_{\text{net}}}{\partial\underline{\theta}_j} = \sum_{i=0}^N 
    \left[ 
        \underbrace{
            \frac{\partial P_i}{\partial\underline{\theta}_j} A_i(\underline{\theta}_j)\underline{n}_i(\underline{\theta}_j)
        }_\text{pressure sensitivity}
        \: + \:
        \underbrace{
            P_i(\underline{\theta}_j) \frac{\partial A_i}{\partial\underline{\theta}_j} \underline{n}_i(\underline{\theta}_j)
        }_\text{area sensitivity}
        \: + \:
        \underbrace{
            P_i(\underline{\theta}_j) A_i(\underline{\theta}_j)\frac{\partial\underline{n}_i}{\partial\underline{\theta}_j}
        }_\text{normal sensitivity}
    \right]
$$


These sensitivities can also be converted into non-dimensional 
force coefficients by dividing through by 
$\frac{1}{2} \, \rho_\infty \, V_\infty^2 \, A_{ref} $.

