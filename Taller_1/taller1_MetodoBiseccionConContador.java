package Taller_1;

import java.util.ArrayList;
import java.util.List;

public class taller1_MetodoBiseccionConContador {
    @FunctionalInterface 
    //Cualquier función que reciba un double y devuelva un double puede ser usada
    public interface Funcion{
        double evaluar(double theta, double orden_Contador);
    }

    public static List<Double> metodo_BiseccionConContador(Funcion funcion, double orden_Contador ,double inicio, double fin){
        /*Sirve para hallar raice de funciones por método numérico */
        double incremento = 0.01; //Incremento
        boolean condition = true; //Condición para el ciclo while
        double theta_izquierda = inicio; //evaluación izquierda
        double theta_derecha = inicio+incremento; //evaluación derecha
        List <Double> raices = new ArrayList<>(); //Lista vacía para almacenar las raíces encontradas 
        int tamaño_AnteriorLista = raices.size(); //Variable para almacenar el tamaño de la lista
        int tamaño_ActualLista; //Variable para almacenar el tamaño de la lista
        while (condition){
            double resultado_izquierda = funcion.evaluar(theta_izquierda, orden_Contador); //Evaluación izquierda
            double resultado_derecha = funcion.evaluar(theta_derecha, orden_Contador); //Evaluación derecha
            if (resultado_derecha*resultado_izquierda < 0 && Math.abs(resultado_derecha+resultado_izquierda) < 1){
                //Se encuentra un cambio de signo, por lo tanto hay una raíz entre estos valores
                double raiz = (theta_derecha+theta_izquierda)/2; //Se propone que esta es la raíz
                double resultado_raiz = funcion.evaluar(raiz, orden_Contador); //Se evalua la función en la raíz propuesta
                if(resultado_izquierda*resultado_raiz < 0){
                    //Significa que la raíz está entre el límite izquierdo y la raíz propuesta
                    theta_derecha = raiz; //Se ajusta el límite derecho y se repite desde el paso 2
                }else if (resultado_izquierda*resultado_raiz > 0) {
                    //Significa que la raíz está entre el límite derecho y la raíz propuesta
                    theta_izquierda = raiz; //Se ajusta el límite izquierdo y se repite desde el paso 2
                }
                if(Math.abs(resultado_raiz) < 1E-7){
                    //Se encontró la raíz exacta, entonces la variable raíz es la solución 
                    raices.add(raiz); //Se agrega la raíz a la lista
                    raices.add(orden_Contador); //Se agrega el orden de la interferencia a la lista
                    theta_izquierda =raiz+incremento; //Se ajusta el límite izquierdo
                    theta_derecha = theta_izquierda + incremento; //Se ajusta el límite derecho  
                }
            }else{
                //No se encontró un cambio de signo, por lo tanto se ajustan los límites, se incrementa el ángulo y se repite el proceso
                theta_izquierda += incremento; //Se incrementa el límite izquierdo
                theta_derecha += incremento; //Se incrementa el límite derecho
            }
            if (theta_derecha >= fin){
                //Se ha llegado al límite derecho, por lo tanto se detiene el ciclo
                tamaño_ActualLista = raices.size(); //Se obtiene el tamaño actual de la lista para comparar
                if (tamaño_ActualLista == tamaño_AnteriorLista){
                    //No se encontró ninguna raíz, por lo tanto se detiene el bucle pues ya no hay sentido en evaluar más ordenes
                    condition = false; //Se cambia la variable de control para salir del bucle
                }else{
                orden_Contador++; //Se incrementa el orden de la interferencia
                theta_izquierda = inicio; //Se reinicia el límite izquierdo
                theta_derecha = theta_izquierda + incremento; //Se reinicia el límite derecho
                tamaño_AnteriorLista = raices.size(); //Se actualiza el tamaño anterior de la lista
                }
            }
        }
        return raices; //Se devuelve la lista de raíces encontradas
    }

}
