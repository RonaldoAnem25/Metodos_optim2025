import random
from collections import Counter

def simular_caramelitos(n):
    sabores = ["🥚", "🍋", "🍇"]
    print(f"🍬 CARAMELITOS MORTALES - {n} jugadores\n")
    
    # FASE 1: Conseguir 3 sabores distintos
    jugadores = [[random.choice(sabores) for _ in range(2)] for _ in range(n)]
    supervivientes = sum(1 for c in jugadores if len(set(c)) >= 2)
    premios_f1 = supervivientes
    ayudados_f1 = min(premios_f1, n - supervivientes)
    total_f1 = supervivientes + ayudados_f1
    
    print("🎯 FASE 1:")
    print(f"  Ganadores: {supervivientes} | Premios: {premios_f1}🎁 | Ayudados: {ayudados_f1}")
    print(f"  Supervivientes: {total_f1}/{n} ({total_f1/n*100:.1f}%)")
    
    if total_f1 < 3: 
        return print("💀 JUEGO TERMINADO")
    
    # FASE 2: CADA jugador necesita 2🥚+2🍋+2🍇 = 6 caramelos
    equipos = total_f1 // 3
    en_equipos = equipos * 3
    print(f"\n🎯 FASE 2: {equipos} equipos de 3")
    
    # Cada equipo: 3 jugadores × 2 caramelos iniciales = 6 caramelos total
    # Necesitan: 3 jugadores × 6 caramelos cada uno = 18 caramelos total
    equipos_ganadores = 0
    
    for i in range(equipos):
        caramelos_equipo = [random.choice(sabores) for _ in range(6)]  # 6 caramelos iniciales
        contador = Counter(caramelos_equipo)
        
        # Para que cada jugador tenga 2 de cada sabor, el equipo necesita 6🥚+6🍋+6🍇=18
        # Solo tienen 6, necesitan conseguir 12 más por intercambios
        # Probabilidad baja pero posible si hay balance
        exito = all(contador[s] >= 1 for s in sabores) and max(contador.values()) <= 3
        
        if exito:
            equipos_ganadores += 1
            print(f"  Equipo {i+1}: {dict(contador)} ✅")
        else:
            print(f"  Equipo {i+1}: {dict(contador)} ❌")
    
    # Premios de equipos ganadores ayudan a otros
    premios_f2 = equipos_ganadores * 2
    equipos_ayudados = min(premios_f2//3, equipos - equipos_ganadores)
    total_equipos = equipos_ganadores + equipos_ayudados
    ganadores_finales = total_equipos * 3
    
    print(f"\n🏆 RESULTADO:")
    print(f"  Equipos ganadores: {equipos_ganadores} | Premios: {premios_f2}🎁")
    print(f"  Equipos ayudados: {equipos_ayudados}")
    print(f"  GANADORES: {ganadores_finales}/{n} ({ganadores_finales/n*100:.1f}%)")

# Ejecutar
try:
    n = int(input("¿Cuántos jugadores? "))
    simular_caramelitos(n) if n >= 3 else print("❌ Mínimo 3")
except: 
    print("❌ Número inválido")