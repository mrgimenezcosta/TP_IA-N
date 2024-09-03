import argparse
from tqdm import tqdm
from diezmil import JuegoDiezMil
from template import JugadorEntrenado
from jugador import JugadorAleatorio

def main(politica_filename, verbose):
    puntajes = []
    cant_turnos = []
    puntajes_random = []
    cant_turnos_random = []
    for play in tqdm(range(500)):
        jugador = JugadorEntrenado('qlearning', 'politica_100000.csv')
        juego = JuegoDiezMil(jugador)
        jugador_random = JugadorAleatorio('random')
        juego_random = JuegoDiezMil(jugador_random)
        cantidad_turnos, puntaje_final = juego.jugar(verbose=verbose)
        cantidad_turnos_random, puntaje_final_random = juego_random.jugar(verbose=verbose)
        puntajes.append(puntaje_final)
        cant_turnos.append(cantidad_turnos)
        cant_turnos_random.append(cantidad_turnos_random)
        puntajes_random.append(puntaje_final_random)
    print(f"Cant turnos promedio: {sum(cant_turnos)/len(cant_turnos)}")
    print(f"Cantidad de turnos promedio random: {sum(cant_turnos_random)/len(cant_turnos_random)}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Jugar una partida de 'Diez Mil' con un agente entrenado usando una política predefinida.")

    # Agregar argumentos
    parser.add_argument('-f', '--politica_filename', type=str, help='Archivo con la política entrenada')
    parser.add_argument('-v', '--verbose', action='store_true', help='Activar modo verbose para ver más detalles durante el juego')

    # Parsear los argumentos
    args = parser.parse_args()

    # Llamar a la función principal con los argumentos proporcionados
    main(args.politica_filename, args.verbose)
