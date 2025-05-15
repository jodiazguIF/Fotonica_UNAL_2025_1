package Taller_1;

import java.util.List;
import java.util.ArrayList;

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
        List <Double> n_eff = new ArrayList<>(); //Lista de índices de refracción efectivos
        List <Double> raices = taller1_MetodoBiseccionConContador.metodo_BiseccionConContador( taller1_ModosTrazadoDeRayoz::resultado_FuncionTE, orden_contador, theta_critico, fin);
        //Se obtiene la lista de raíces para el modo TE
        for (int i=0 ; i<raices.size(); i+=2){
            double aux = n_core*Math.sin(raices.get(i)); //Se obtiene el índice de refracción efectivo
            n_eff.add(aux);; 
        }
        for(int i=0 ; i<raices.size(); i+=2){
            raices.set(i, Math.toDegrees(raices.get(i))); //Convierte las raíces a grados
        }
        System.out.println("Thetas para el modo TE: " + raices); //Se imprime la lista de raíces
        System.out.println("n_eff para el modo TE: " + n_eff); //Se imprime la lista de índices de refracción efectivos
        orden_contador = 0; //Reinicia el contador para el modo TM
        raices = taller1_MetodoBiseccionConContador.metodo_BiseccionConContador(taller1_ModosTrazadoDeRayoz::resultado_FuncionTM, orden_contador, theta_critico, fin);
        //Se obtiene la lista de raíces para el modo TM
        n_eff.clear();
        for (int i=0 ; i<raices.size(); i+=2){
            double aux = n_core*Math.sin(raices.get(i)); //Se obtiene el índice de refracción efectivo
            n_eff.add(aux);
        }
        for(int i=0 ; i<raices.size(); i+=2){
            raices.set(i, Math.toDegrees(raices.get(i))); //Convierte las raíces a grados
        }
        System.out.println("Thetas para el modo TM: " + raices); //Se imprime la lista de raíces
        System.out.println("n_eff para el modo TM: " + n_eff); //Se imprime la lista de índices de refracción efectivos
    }
}
