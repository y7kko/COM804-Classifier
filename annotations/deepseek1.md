### 📚 1. Por onde começar: Ajuste manual ou Optuna?

A recomendação prática para quem tem orçamento computacional limitado é:

*   **Fase 1 (Manual e Rápida):** 
    *   Comece com um **baseline simples**.
    *    Ajuste manualmente apenas a **taxa de aprendizado** (Learning Rate) (Mais crítico) 
    *    o **batch size**. 
    * Para a taxa de aprendizado, comece com `1e-3` (Adam) e teste uma ordem de magnitude para cima e para baixo (`1e-4`, `1e-2`).
*   **Fase 2 (Automática e Refinada):** 
    *   depois de ter um baseline estável, use o **Optuna**. 

> **Dica de ouro para o Optuna:** O Optuna tem um recurso chamado **`pruner`** (ex: `MedianPruner`). Ele interrompe automaticamente as tentativas ruins nas primeiras épocas.

### ⏱️ 2. Quantas épocas rodar para ver se a mudança fez efeito?

A literatura e a prática recomendam o uso de **Early Stopping** com uma **"paciência" curta** durante a fase de busca.

*   **A estratégia:** Configure o treino para parar se a acurácia de validação não melhorar por **5 ou 10 épocas**.

### ⚖️ 3. Overfitting: O que focar primeiro?

Se "muita coisa resulta em overfitting", o problema pode não ser os hiperparâmetros, mas o **baseline** ou o **tamanho do modelo**. A literatura sobre tuning sugere focar nesta ordem:

1.  **Regularização (Dropout):** Antes de mexer na arquitetura, tente aumentar o Dropout. A faixa segura costuma ser entre **0,2 e 0,5**.
2.  **Data Augmentation:** A literatura clássica de treinamento de CNNs do zero diz: sem data augmentation, CNNs típicas fazem overfitting muito rapidamente. 
3.  **Complexidade da Arquitetura:** Só depois de esgotar Dropout e Augmentation, tente **reduzir** o número de filtros ou camadas.

### 🏗️ 4. Qual arquitetura escolher? Aplicações específicas têm arquiteturas similares?

Sim, aplicações específicas tendem a ter **famílias de arquiteturas** que funcionam melhor, mas a literatura sobre **treinar do zero** (sem transfer learning) sugere uma abordagem mais humilde.

*   **A verdade sobre "do zero":** Se você não pode usar pesos pré-treinados, comece **simples**. Uma pequena VGG-like (blocos de `Conv2D` + `MaxPooling`) com 3 ou 4 blocos é um excelente ponto de partida. Arquiteturas como LeNet-5 são para tarefas simples, enquanto VGG e ResNet brilham em tarefas complexas, mas exigem mais dados .
*   **Como escolher a família:** Se o seu problema for de **classificação de imagens genéricas**, uma CNN pequena com blocos convolucionais padrão é o "arroz com feijão" que funciona. A literatura sobre tuning de arquitetura em datasets como CIFAR-10 mostra que **`pool_type`, `optimizer`, `activation` e `learning_rate`** são os hiperparâmetros que mais impactam o resultado, enquanto o **número de camadas (`n_layers`) tem impacto limitado** . Ou seja, foque em acertar a ativação (ReLU), o pooling (MaxPooling) e o otimizador (Adam) antes de querer uma rede de 20 camadas.

### 💎 Resumo do Fluxo Metodológico para o seu Trabalho

1.  **Setup:** Defina um dataset de validação fixo. Ative **Data Augmentation** e **Early Stopping** (paciência=10).
2.  **Baseline Manual:** Treine uma CNN pequena (3 blocos) com Adam, LR=`1e-3`, Dropout=`0.5`, Batch=`32` ou `64`. Veja se converge e qual a acurácia de validação.
3.  **Busca com Optuna (Curta):** Rode ~20 a 30 trials com Optuna. Deixe ele variar **Learning Rate, Dropout e Batch Size**. Use o `MedianPruner` para matar trials ruins cedo.
4.  **Análise:** Pegue os melhores hiperparâmetros do Optuna e treine o modelo final por mais épocas (ex: 50-100) com Early Stopping.
5.  **Se der Overfitting:** Aumente o Dropout ou o Data Augmentation. Se der Underfitting, aí sim tente aumentar levemente a complexidade (mais filtros, não necessariamente mais camadas).
