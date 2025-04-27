package Taller_1;

import java.util.List;

public class taller1_ResultadoTrazadoDeRayos {
    public static void main(String[] args) {
        /*Se usará este archivo para poder obtener los resultados con las
         * funciones programadas en esta carpeta
         */
        double n_clad = 1.0; //Índice de refracción del revestimiento
        double n_core = 1.5; //Índice de refracción del núcleo 
        double fin = Math.PI/2; //Límite superior del ángulo
        double theta_critico = (double)(Math.asin(n_clad/n_core)); //Ángulo crítico para iniciar con el cálculo numérico
        double orden_contador = 0; //Orden del modo de propagación
        List <Double> raices = taller1_MetodoBiseccionConContador.metodo_BiseccionConContador( taller1_ModosTrazadoDeRayoz::resultado_FuncionTE, orden_contador, theta_critico, fin);
        //Se obtiene la lista de raíces para el modo TE
        System.out.println("Raíces para el modo TE: " + raices); //Se imprime la lista de raíces
        orden_contador = 0; //Reinicia el contador para el modo TM
        raices = taller1_MetodoBiseccionConContador.metodo_BiseccionConContador(taller1_ModosTrazadoDeRayoz::resultado_FuncionTM, orden_contador, theta_critico, fin);
        //Se obtiene la lista de raíces para el modo TM
        System.out.println("Raíces para el modo TM: " + raices); //Se imprime la lista de raíces
    }
}
