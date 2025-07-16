import pandas as pd
import numpy as np
from scipy.optimize import minimize
import time
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')

class PortfolioOptimizer:
    """Optimizador de portafolio para OptimaBattle Arena"""

    def __init__(self, data_path: str):
        """Inicializa el optimizador con los datos del archivo Excel"""
        self.df = pd.read_excel(data_path)
        self.n_assets = len(self.df)
        self.budget = 1_000_000
        self.lambda_risk = 0.5
        self.start_time = time.time()

        # Convertir columnas a arrays
        self.returns = self.df['retorno_esperado'].values / 100
        self.volatilities = self.df['volatilidad'].values / 100
        self.betas = self.df['beta'].values
        self.prices = self.df['precio_accion'].values
        self.min_investments = self.df['min_inversion'].values
        self.sectors = self.df['sector'].values
        self.liquidity_scores = self.df['liquidez_score'].values

    def objective_function(self, weights: np.ndarray) -> float:
        """Maximizar utilidad esperada: retorno - λ * riesgo"""
        portfolio_return = np.dot(weights, self.returns)
        portfolio_risk = np.dot(weights ** 2, self.volatilities ** 2)
        return -(portfolio_return - self.lambda_risk * portfolio_risk)

    def _calculate_weights(self, shares: np.ndarray) -> np.ndarray:
        """Calcula los pesos del portafolio según acciones"""
        values = shares * self.prices
        total = np.sum(values)
        return values / total if total > 0 else np.zeros(self.n_assets)

    def get_constraints(self) -> List[Dict]:
        """Construye las restricciones del modelo"""
        constraints = []

        # 1. Presupuesto
        constraints.append({
            'type': 'ineq',
            'fun': lambda x: self.budget - np.dot(x[:self.n_assets], self.prices)
        })

        # 2. Diversificación sectorial <= 30%
        for sector in np.unique(self.sectors):
            mask = (self.sectors == sector).astype(float)
            constraints.append({
                'type': 'ineq',
                'fun': lambda x, m=mask: 0.30 - np.dot(self._calculate_weights(x[:self.n_assets]), m)
            })

        # 3. Mínimo 5 activos seleccionados
        constraints.append({
            'type': 'ineq',
            'fun': lambda x: np.sum(x[self.n_assets:]) - 5
        })

        # 4. Riesgo sistemático (beta)
        constraints.append({
            'type': 'ineq',
            'fun': lambda x: 1.2 - np.dot(self.betas, self._calculate_weights(x[:self.n_assets]))
        })

        # 5. Inversión mínima y lógica (vincular xi y yi)
        for i in range(self.n_assets):
            price_i = self.prices[i]
            min_inv = self.min_investments[i]
            M = self.budget / price_i

            constraints.append({
                'type': 'ineq',
                'fun': lambda x, i=i: x[i] * price_i - min_inv * x[self.n_assets + i]
            })
            constraints.append({
                'type': 'ineq',
                'fun': lambda x, i=i: M * x[self.n_assets + i] - x[i]
            })

        return constraints

    def get_bounds(self) -> List[Tuple[float, float]]:
        """Límites para cada variable: acciones y binarios"""
        bounds = [(0, self.budget / p) for p in self.prices]
        bounds += [(0, 1)] * self.n_assets
        return bounds

    def optimize(self) -> Dict:
        """Ejecuta la optimización principal"""
        x0 = np.zeros(2 * self.n_assets)

        # Inicialización inteligente
        scores = (self.returns / self.volatilities) * (self.liquidity_scores / 10)
        top = np.argsort(scores)[-10:]
        for i in top:
            x0[i] = self.min_investments[i] / self.prices[i]
            x0[self.n_assets + i] = 1

        result = minimize(
            fun=lambda x: self.objective_function(self._calculate_weights(x[:self.n_assets])),
            x0=x0,
            method='SLSQP',
            bounds=self.get_bounds(),
            constraints=self.get_constraints(),
            options={'maxiter': 1000, 'ftol': 1e-9, 'disp': False}
        )

        if result.success:
            x_opt = result.x
            shares = x_opt[:self.n_assets]
            binaries = np.round(x_opt[self.n_assets:])
            shares *= binaries
            weights = self._calculate_weights(shares)

            ret = np.dot(weights, self.returns)
            vol = np.sqrt(np.dot(weights ** 2, self.volatilities ** 2))
            beta = np.dot(weights, self.betas)
            elapsed = time.time() - self.start_time
            score = self._calculate_score(ret, vol, elapsed)

            return {
                'success': True,
                'solution': self._create_solution(shares, weights, binaries),
                'metrics': {
                    'retorno_esperado': ret * 100,
                    'volatilidad': vol * 100,
                    'beta': beta,
                    'utilidad': ret - self.lambda_risk * vol ** 2,
                    'n_activos': int(np.sum(binaries)),
                    'inversion_total': np.dot(shares, self.prices),
                    'tiempo_ejecucion': elapsed,
                    'puntaje': score
                },
                'verificacion': self._verify_constraints(shares, weights, binaries)
            }

        else:
            return {'success': False, 'message': 'Optimización fallida', 'reason': result.message}

    def _create_solution(self, shares, weights, binaries) -> pd.DataFrame:
        """Genera el portafolio como DataFrame"""
        selected = binaries > 0.5
        return pd.DataFrame({
            'activo_id': self.df.loc[selected, 'activo_id'].values,
            'n_acciones': shares[selected].astype(int),
            'precio': self.prices[selected],
            'inversion': (shares[selected] * self.prices[selected]).round(2),
            'peso': (weights[selected] * 100).round(2),
            'retorno_esperado': self.returns[selected] * 100,
            'volatilidad': self.volatilities[selected] * 100,
            'beta': self.betas[selected],
            'sector': self.sectors[selected],
            'liquidez': self.liquidity_scores[selected]
        })

    def _verify_constraints(self, shares, weights, binaries) -> Dict:
        """Verifica restricciones y devuelve un resumen"""
        total_investment = np.dot(shares, self.prices)
        verif = {
            'presupuesto': {
                'cumple': total_investment <= self.budget,
                'valor': total_investment,
                'limite': self.budget,
                'holgura': self.budget - total_investment
            },
            'n_activos': {
                'cumple': np.sum(binaries) >= 5,
                'valor': int(np.sum(binaries)),
                'limite': 5
            },
            'beta': {
                'cumple': (beta := np.dot(weights, self.betas)) <= 1.2,
                'valor': beta,
                'limite': 1.2
            },
            'inversiones_minimas': {
                'cumple': all(
                    shares[i] * self.prices[i] >= self.min_investments[i]
                    for i in range(self.n_assets) if binaries[i] > 0.5
                ),
                'violaciones': sum(
                    shares[i] * self.prices[i] < self.min_investments[i]
                    for i in range(self.n_assets) if binaries[i] > 0.5
                )
            },
            'diversificacion_sectorial': {
                f'sector_{s}': {
                    'cumple': (sw := np.sum(weights[self.sectors == s])) <= 0.30,
                    'valor': sw,
                    'limite': 0.30
                }
                for s in np.unique(self.sectors)
            }
        }
        return verif

    def _calculate_score(self, retorno, volatilidad, tiempo) -> float:
        """Calcula el puntaje final del portafolio"""
        Fr = 1.0
        Ft = 1.5 if tiempo < 900 else 1.2 if tiempo < 1200 else 1.0
        score = 1000 * (retorno - 0.5 * volatilidad) * Fr * Ft
        return round(score, 2)

def main():
    print("=== OPTIMABATTLE ARENA - OPTIMIZADOR DE PORTAFOLIO ===\n")

    optimizer = PortfolioOptimizer('Ronda1.xlsx')

    print(f"Activos cargados: {optimizer.n_assets}")
    print(f"Presupuesto: S/. {optimizer.budget:,.2f}")
    print("Iniciando optimización...\n")

    results = optimizer.optimize()

    if results['success']:
        print("✅ Optimización exitosa\n")
        m = results['metrics']
        print(f"Retorno esperado: {m['retorno_esperado']:.2f}%")
        print(f"Volatilidad: {m['volatilidad']:.2f}%")
        print(f"Beta: {m['beta']:.3f}")
        print(f"Utilidad: {m['utilidad']:.4f}")
        print(f"Activos seleccionados: {m['n_activos']}")
        print(f"Inversión total: S/. {m['inversion_total']:,.2f}")
        print(f"Tiempo: {m['tiempo_ejecucion']:.2f}s")
        print(f"PUNTAJE FINAL: {m['puntaje']:.2f}")

        print("\n📈 Portafolio:")
        print(results['solution'].to_string(index=False))

        print("\n✔️ Verificación de restricciones:")
        ver = results['verificacion']
        print(f"• Presupuesto: {'✓' if ver['presupuesto']['cumple'] else '✗'} "
              f"(Usado: S/. {ver['presupuesto']['valor']:,.2f})")
        print(f"• N° de activos: {'✓' if ver['n_activos']['cumple'] else '✗'}")
        print(f"• Beta: {'✓' if ver['beta']['cumple'] else '✗'}")
        print(f"• Inversión mínima: {'✓' if ver['inversiones_minimas']['cumple'] else '✗'}")
        print("• Diversificación sectorial:")
        for s, info in ver['diversificacion_sectorial'].items():
            print(f"  - {s}: {'✓' if info['cumple'] else '✗'} ({info['valor']*100:.1f}%)")

        results['solution'].to_excel('solucion_portafolio.xlsx', index=False)
        print("\n💾 Solución guardada como 'solucion_portafolio.xlsx'")
    else:
        print("❌ Optimización fallida:")
        print(results['reason'])

if __name__ == '__main__':
    main()
