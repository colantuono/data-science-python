### Transformações

- Logaritmica: usada para reduzir a assimetria da distribuição, especialmente util para dados que tem um assimetria positiva. 
  $$y = log_{10}(x)$$

- Potencia: util para dados que tem uma assimetria negativa (o valor da potencia depende da assimetria)
  $$y=x^{pow}$$

- Raiz Quadrada: para dados com assimetria positiva moderada
  $$y=\sqrt{x}$$

- Exponencial: para dados com assimetria negativa
  $$y=e^{x}$$

- Box-Cox: transformação paramtrica que visa melhorar a normalidade dos dados
  $$y=frac{(x^{\epsilon}-1)}{\epsilon}$$

- Yeo-Johnson: Extensão da Box-Cox, que suporta valores zero e negativos:
  For strictly positive values (x > 0): (x^(λ) - 1) / λ + 1
  For strictly negative values (x < 0): (-(-x)^(2-λ) - 1) / (2-λ) + 1
  For x = 0, the transformation is undefined