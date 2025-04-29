package Taller_1;
import java.util.ArrayList;
import java.util.List;

public class taller1_MetodoBiseccion {
    @FunctionalInterface 
    //Cualquier función que reciba un double y devuelva un double puede ser usada
    public interface Funcion{
        double evaluar(double theta);
    }

    public static List<Double> metodo_Biseccion(Funcion funcion, double inicio, double fin){
        /*Sirve para hallar raice de funciones por método numérico */
        double incremento = 0.01; //Incremento
        boolean condition = true; //Condición para el ciclo while
        double theta_izquierda = inicio; //evaluación izquierda
        double theta_derecha = inicio+incremento; //evaluación derecha
        List <Double> raices = new ArrayList<>(); //Lista vacía para almacenar las raíces encontradas 
        while (condition){
            double resultado_izquierda = funcion.evaluar(theta_izquierda); //Evaluación izquierda
            double resultado_derecha = funcion.evaluar(theta_derecha); //Evaluación derecha
            if (resultado_derecha*resultado_izquierda < 0 && Math.abs(resultado_derecha+resultado_izquierda) < 1){
                //Se encuentra un cambio de signo, por lo tanto hay una raíz entre estos valores
                double raiz = (theta_derecha+theta_izquierda)/2; //Se propone que esta es la raíz
                double resultado_raiz = funcion.evaluar(raiz); //Se evalua la función en la raíz propuesta
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
                    theta_izquierda =raiz+incremento; //Se ajusta el límite izquierdo
                    theta_derecha = theta_izquierda + incremento; //Se ajusta el límite derecho  
                }
            }else{
                //No se encontró un cambio de signo, por lo tanto se ajustan los límites, se incrementa el ángulo y se repite el proceso
                theta_izquierda += incremento; //Se incrementa el límite izquierdo
                theta_derecha += incremento; //Se incrementa el límite derecho
            }
            if (theta_derecha >= fin){
                //Se ha llegado al límite derecho, por lo tanto se termina el ciclo
                condition = false; //Se cambia la condición para salir del ciclo
            }
        }
        return raices; //Se devuelve la lista de raíces encontradas
    }
}
