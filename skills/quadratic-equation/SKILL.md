---
name: quadratic-equation
description: Use when the user asks for the roots of a quadratic equation ax^2 + bx + c = 0.
---

# Quadratic equation

The values a, b, and c come from the user's message. Do not read a data file to find them.

1. Use the calculator to compute the discriminant: b**2 - 4*a*c.
2. If the discriminant is negative, say there are no real roots.
3. If it is zero, there is one real root: -b / (2*a). Use the calculator.
4. If it is positive, the roots are (-b + sqrt(discriminant)) / (2*a) and (-b - sqrt(discriminant)) / (2*a). Use **0.5 for the square root, and use the calculator for each step.
