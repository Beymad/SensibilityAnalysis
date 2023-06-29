import numpy as np

def HarrisEOQ(inputM, inputS, inputC,IAHEOQ):                       # Exemplo de Borgonovo e Plischke, Sensitivity analysis: A review of recent advances
                                                                    # inputM      -> Número de unidades por mês / Saída do produto por mês 
    modelResult = np.sqrt((IAHEOQ*inputM*inputC)/inputS)            # inputS      -> Custo inicial de um pedido
                                                                    # inputC      -> Custo unitário de uma unidade / de armazenamento por unidade
    return modelResult                                              # modelResult -> Quantidade Otima de Pedido / Quanto que se deve pedir de produto de uma vez só para minimizar custos
