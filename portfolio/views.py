# en views.py
from django.http import HttpResponse, HttpResponseRedirect
from django.shortcuts import render
from django.contrib import messages
import pandas as pd
import os
import yfinance as yf
from sqlalchemy import create_engine
from django.core.paginator import Paginator
from django.conf import settings
from django.shortcuts import render, redirect
import pandas as pd
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime, timedelta

# Crear la conexión a la base de datos
engine = create_engine('sqlite:///db.sqlite3')

def obtener_informacion_yfinance(symbol):
    try:
        ticker = yf.Ticker(symbol)
        last_price = ticker.info['regularMarketPreviousClose']
        last_price = "{:.2f}".format(last_price)
        informacion = {
            'symbol': symbol,
            'name': ticker.info['longName'],
            'last_price': last_price,
            'currency' : ticker.info['currency']
        }
        return informacion
    except Exception as e:
        return None

def busqueda(request):
    results = []
    paginator = []
    page = []
    total_pages = None
    search_text = request.POST.get('search_text')
    search_type = request.POST.get('search_type')

    if request.method == 'POST':
        if search_type == 'Símbolo':
            query = f'SELECT * FROM "symbols" WHERE "symbol" LIKE "%{search_text}%"'
        elif search_type == 'Nombre':
            query = f'SELECT * FROM "symbols" WHERE "name" LIKE "%{search_text}%"'
        request.session['query'] = query
    else:
        if request.GET.get('page') == None:
            query = None
            request.session['query'] = query

    page_number = int(request.GET.get('page', 1))

    try:
        if request.session['query']:
            query = request.session['query']

        # Obtener el total de registros para calcular el número total de páginas
        if not query == None:
            # Obtener el número de página solicitado
            page_number = int(request.GET.get('page', 1))
            per_page = 10
            offset = (page_number - 1) * per_page

            # Ajustar la consulta para paginación
            result_page = pd.read_sql_query(query, engine)
            paginated_query = f'{query} LIMIT {per_page} OFFSET {offset}'
            result = pd.read_sql_query(paginated_query, engine)
            
            if not result.empty:
                symbols = result['symbol'].tolist()
                informacion = []
                for symbol in symbols:
                    info = obtener_informacion_yfinance(symbol)
                    if info:
                        informacion.append(info)
                
                # Paginar los resultados
                paginator = Paginator(result_page, per_page)
                total_pages = paginator.num_pages
                page = paginator.page(page_number)
                results = informacion
            else:
                total_pages=-1

            return render(request, 'busqueda.html', {'results': results, 'total_pages': total_pages, 'entity': page, 'paginator': paginator})
        else:
            return render(request, 'busqueda.html', {'results': None})
    except Exception as e:
        print(f"Error: {e}")
        return render(request, 'busqueda.html', {'results': None})

def generar_portafolio(request):
    try:
        if request.session['portafolio']:
            portafolio = request.session['portafolio']
        return render(request, 'portafolio.html', {'results': portafolio})
    except Exception as e:
        messages.error(request, "Es necesario agregar activos al portafolio")
        return render(request, 'portafolio.html', {'results': None})

def agregar_fila(request):
    if request.method == 'POST':
        symbol = request.POST.get('symbol')
        name = request.POST.get('name')
        currency = request.POST.get('currency')
        last_price = request.POST.get('last_price')
        
        portafolio = request.session.get('portafolio', [])

        # Comprobar si el símbolo ya existe en el portafolio
        simbolo_existe = any(entry['symbol'] == symbol for entry in portafolio)

        if simbolo_existe:
            messages.error(request, f"{symbol} ya está en el portafolio.")
        else:
            nuevo_registro = {
                'symbol': symbol,
                'name': name,
                'last_price': last_price,
                'currency': currency
            }
            portafolio.append(nuevo_registro)
            request.session['portafolio'] = portafolio
            messages.success(request, f"{symbol} agregado al portafolio.")

        return render(request, 'portafolio.html', {'results': portafolio})

def eliminar_fila(request):
    if request.method == 'POST':
        symbol = request.POST.get('symbol')
        
        portafolio = request.session.get('portafolio', [])
        portafolio = [entry for entry in portafolio if entry['symbol'] != symbol]
        request.session['portafolio'] = portafolio

        messages.success(request, f"{symbol} eliminado del portafolio.")
        return render(request, 'portafolio.html', {'results': portafolio})

# Función para obtener datos históricos de yfinance
def obtener_historico(symbols):
    try:
        end_date = datetime.now()
        end_date_str = end_date.strftime('%Y-%m-%d')
        start_date = (datetime.now() - timedelta(days=5*365) - timedelta(days=30)).strftime('%Y-%m-%d')
        data = yf.download(symbols, start=start_date, end=end_date_str, interval='1mo')['Adj Close']
        data.to_csv('prueba-portafolio.csv')
        return data
    except Exception as e:
        print(f"Error: {e}")
        return None
    
# Función para obtener datos históricos de US Bonos
def obtener_historico_bonos():
    try:
        end_date = datetime.now()
        end_date_str = end_date.strftime('%Y-%m-%d')
        start_date = (datetime.now() - timedelta(days=5*365) - timedelta(days=30)).strftime('%Y-%m-%d')
        data = yf.download('^TNX', start=start_date, end=end_date_str, interval='1mo')['Close']
        tasa_libre_de_riesgo = data['^TNX'].mean() / 100
        print('aqui bon')
        return tasa_libre_de_riesgo
    except Exception as e:
        return None
    
# Función para calcular el ratio de Sharpe
def calcular_ratio_sharpe(retorno, desv_std):
    try:
        end_date = datetime.now()
        end_date_str = end_date.strftime('%Y-%m-%d')
        start_date = (datetime.now() - timedelta(days=5*365) - timedelta(days=30)).strftime('%Y-%m-%d')
        data = yf.download('^TNX', start=start_date, end=end_date_str, interval='1mo')['Adj Close']
        tasa_libre_de_riesgo = data.mean() / 100        
        if tasa_libre_de_riesgo is None:
            tasa_libre_de_riesgo = 0.0
        ratio_sharpe = (retorno - tasa_libre_de_riesgo) / desv_std
        return ratio_sharpe
    except Exception as e:
        return None


# Función para generar el gráfico de la frontera eficiente
def generar_grafico_frontera_eficiente(mean_returns, cov_matrix, num_portfolios=20000):
    results = np.zeros((4, num_portfolios))
    weights_record = []
    for i in range(num_portfolios):
        weights = np.random.random(len(mean_returns))
        weights /= np.sum(weights)
        portfolio_return = np.dot(weights, mean_returns)
        portfolio_stddev = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
        results[0,i] = portfolio_return
        results[1,i] = portfolio_stddev
        results[2,i] = results[0,i] / results[1,i]
        weights_record.append(weights)
    plt.figure(figsize=(10, 7))
    plt.scatter(results[1,:], results[0,:], c=results[2,:], cmap='YlGnBu', marker='o')
    plt.title('Frontera Eficiente')
    plt.xlabel('Riesgo (Desviación Estándar)')
    plt.ylabel('Retorno')

    # Definir el path completo de la imagen
    image_path = os.path.join(settings.STATICFILES_DIRS[0], 'images', 'frontera_eficiente.png')

    # Guardar la imagen en la carpeta images/
    plt.savefig(image_path)
    plt.close()

    return '/images/frontera_eficiente.png'  # Devuelve la ruta relativa

def calcular_portafolio_equitativo(mean_returns, cov_matrix, num_assets):
    weights = np.ones(num_assets) / num_assets
    retorno_portafolio = np.dot(weights, mean_returns)
    varianza_portafolio = np.dot(weights.T, np.dot(cov_matrix, weights))
    desviacion_portafolio = np.sqrt(varianza_portafolio)
    matriz_markowitz = np.outer(weights, weights) * cov_matrix
    return retorno_portafolio, varianza_portafolio, desviacion_portafolio, weights, matriz_markowitz

def calcular_portafolio_minima_varianza(mean_returns, cov_matrix, num_assets):
    def portafolio_varianza(weights):
        return np.dot(weights.T, np.dot(cov_matrix, weights))

    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    bounds = tuple((0, 1) for asset in range(num_assets))
    initial_weights = np.ones(num_assets) / num_assets

    result = minimize(portafolio_varianza, initial_weights, method='SLSQP', bounds=bounds, constraints=constraints)
    weights = result.x
    retorno_portafolio = np.dot(weights, mean_returns)
    varianza_portafolio = np.dot(weights.T, np.dot(cov_matrix, weights))
    desviacion_portafolio = np.sqrt(varianza_portafolio)
    matriz_markowitz = np.outer(weights, weights) * cov_matrix
    return retorno_portafolio, varianza_portafolio, desviacion_portafolio, weights, matriz_markowitz

def calcular_portafolio_maximo_retorno(mean_returns, cov_matrix, num_assets):
    def portafolio_retorno(weights):
        return -np.dot(weights, mean_returns)

    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    bounds = tuple((0, 1) for asset in range(num_assets))
    initial_weights = np.ones(num_assets) / num_assets

    result = minimize(portafolio_retorno, initial_weights, method='SLSQP', bounds=bounds, constraints=constraints)
    weights = result.x
    retorno_portafolio = np.dot(weights, mean_returns)
    varianza_portafolio = np.dot(weights.T, np.dot(cov_matrix, weights))
    desviacion_portafolio = np.sqrt(varianza_portafolio)
    matriz_markowitz = np.outer(weights, weights) * cov_matrix
    return retorno_portafolio, varianza_portafolio, desviacion_portafolio, weights, matriz_markowitz

def calcular_portafolio_personalizado(mean_returns, cov_matrix, custom_weights):
    weights = np.array([custom_weights[symbol] for symbol in mean_returns.index])
    retorno_portafolio = np.dot(weights, mean_returns)
    varianza_portafolio = np.dot(weights.T, np.dot(cov_matrix, weights))
    desviacion_portafolio = np.sqrt(varianza_portafolio)
    matriz_markowitz = np.outer(weights, weights) * cov_matrix
    return retorno_portafolio, varianza_portafolio, desviacion_portafolio, weights, matriz_markowitz

def analisis(request):
    if request.method == 'POST':
        matriz_assets = request.session.get('portafolio', [])
        analysis_type = request.POST.get('analysis_type')
        request.session['analysis_type'] = analysis_type
        action = request.POST.get('action')
    elif request.method == 'GET':
        matriz_assets = request.session.get('portafolio', [])
        analysis_type = request.session.get('analysis_type')
        action = 'procesar'
        if(analysis_type == None):
            analysis_type = 'Equitativo'
        portafolio = request.session.get('portafolio', [])
        if portafolio == [] or len(portafolio) < 2:
            portafolio == []
            messages.error(request, "Es necesario agregar activos al portafolio")
            return render(request, 'portafolio.html', {'results': portafolio})
        
    if action == 'borrar':
        # Borra todo el contenido del portafolio en la sesión
        request.session['portafolio'] = []
        request.session['analysis_type'] = 'Equitativo'
        portafolio = []
        messages.success(request, "El portafolio ha sido borrado exitosamente.")
        return render(request, 'portafolio.html', {'results': portafolio})
    elif action == 'procesar':
        try:
            assets = [item['symbol'] for item in matriz_assets]
        except KeyError:
            assets = []
        request.session['matriz-assets'] = assets

        historico = obtener_historico(assets)

        if len(assets) > 1:
            returns = historico.pct_change(fill_method=None).dropna()
            mean_returns = returns.mean()
            cov_matrix = returns.cov()
            num_assets = len(mean_returns)

            if analysis_type == 'Equitativo':
                p_type = 'Equitativo'
                request.session['analysis_type'] = 'Equitativo'
                retorno_portafolio, varianza_portafolio, desviacion_portafolio, weights, matriz_markowitz = calcular_portafolio_equitativo(mean_returns, cov_matrix, num_assets)
            elif analysis_type == 'Menor':
                p_type = 'Mínimo Riesgo'
                request.session['analysis_type'] = 'Menor'
                retorno_portafolio, varianza_portafolio, desviacion_portafolio, weights, matriz_markowitz = calcular_portafolio_minima_varianza(mean_returns, cov_matrix, num_assets)
            elif analysis_type == 'Mayor':
                p_type = 'Máximo Retorno'
                request.session['analysis_type'] = 'Mayor'
                retorno_portafolio, varianza_portafolio, desviacion_portafolio, weights, matriz_markowitz = calcular_portafolio_maximo_retorno(mean_returns, cov_matrix, num_assets)
            elif analysis_type == 'Otro':
                p_type = 'Otro'
                weights = {}
                for key in request.POST:
                    if key.startswith('weights_'):
                        symbol = key.split('_')[1]
                        weight = float(request.POST[key])/100
                        weights[symbol] = weight
                request.session['custom_weights'] = weights  # Guardar los pesos personalizados en la sesión
                request.session['analysis_type'] = 'Otro'
                if sum(weights.values()) == 1:
                    retorno_portafolio, varianza_portafolio, desviacion_portafolio, weights, matriz_markowitz = calcular_portafolio_personalizado(mean_returns, cov_matrix, weights)
                else:
                    messages.error(request, "La suma de todas las ponderaciones debe ser 100%")
                    return HttpResponseRedirect('/portafolio/')

            # Formatear los valores como porcentajes
            formatted_weights = [f"{w * 100:.2f}%" for w in weights]
            formatted_retorno_portafolio = f"{retorno_portafolio * 100:.2f}%"
            formatted_varianza_portafolio = f"{varianza_portafolio * 100:.2f}%"
            formatted_desviacion_portafolio = f"{desviacion_portafolio * 100:.2f}%"
            formatted_markowitz_matrix = matriz_markowitz * 100  # Convertir a porcentaje
            formatted_markowitz_matrix = formatted_markowitz_matrix.map(lambda x: f"{x:.2f}%")

            # Calcular el ratio de Sharpe
            ratio_sharpe = calcular_ratio_sharpe(retorno_portafolio,desviacion_portafolio)
            formatted_ratio_sharpe = f"{ratio_sharpe.mean():.2f}"

            stats = pd.DataFrame({
                'EST': ['Promedio', 'Varianza', 'Desv. Est.'],
                'Valores': [
                    [f"{mean * 100:.2f}%" for mean in mean_returns],
                    [f"{var * 100:.3f}%" for var in returns.var()],
                    [f"{std * 100:.3f}%" for std in returns.std()],
                ]
            })

            stats_list = stats.values.tolist()
            symbols = historico.columns.tolist()
            num_columns = len(symbols) + 1

            formatted_cov_matrix = cov_matrix.map(lambda x: f"{x * 100:.2f}%")  # Convertir a porcentaje
            cov_matrix_list = formatted_cov_matrix.values.tolist()
            combined_list = list(zip(symbols, cov_matrix_list))

            markowitz_matrix_list = formatted_markowitz_matrix.values.tolist()
            combined_list_markowitz = list(zip(symbols, markowitz_matrix_list))

            return render(request, 'analisis.html', {
                'stats': stats_list,
                'symbols': symbols,
                'num_columns': num_columns,
                'cov_matrix': cov_matrix_list,
                'combined_list': combined_list,
                'analysis_type': p_type,
                'retorno_portafolio': formatted_retorno_portafolio,
                'varianza_portafolio': formatted_varianza_portafolio,
                'desviacion_portafolio': formatted_desviacion_portafolio,
                'sharpe_portafolio': formatted_ratio_sharpe,
                'weights': formatted_weights,
                'markowitz_matrix': combined_list_markowitz
            })
        else:
            messages.error(request, "Es necesario agregar más activos al portafolio")
            return HttpResponseRedirect('/portafolio/')
