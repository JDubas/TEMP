from flask import Flask, request, jsonify
import numpy as np
import pickle

app = Flask(__name__)

# Modelo de exemplo - substitua isso pelo seu modelo treinado
# Aqui estamos carregando o modelo já treinado que está no arquivo JobLib
with open('C:/Users/Ferreira/Desktop/modelo_treinado_gafesp.pickle', 'rb') as f:
    modelo = pickle.load(f)

# Rota para receber os dados e fazer previsões
@app.route('/prever', methods=['GET'])
def prever():
    # Obter parâmetros da solicitação GET
    parametro1 = float(request.args.get('comp_abd'))
    parametro2 = float(request.args.get('comp_ant'))

    # Fazer previsões usando o modelo (substitua por suas próprias lógicas)
    entrada = np.array([[parametro1, parametro2]])
    resultado = modelo.predict(entrada)

    # Retornar o resultado como JSON
    return jsonify({'previsao': resultado.tolist()})

if __name__ == '__main__':
    print("Servidor Flask em execução")
    # Executar o aplicativo Flask
    app.run(debug=True)
