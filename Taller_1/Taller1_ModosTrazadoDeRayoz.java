package Taller_1;
import java.util.ArrayList;
import java.util.List;

public class Taller1_ModosTrazadoDeRayoz{
    public static double resultado_FuncionTE(double theta, double n_core, double n_clad, double long_Onda, double altura_fibra ,double m){
        /*Esta función recibe los argumentos necesarios para calcular la suma de todos los cambios de fase en ondas TE */
        double Ecuacion = (double)(((4 * Math.PI * n_core*altura_fibra*Math.cos(theta))/(long_Onda)) 
        - 4*Math.atan(((n_clad/(n_core*Math.cos(theta))))*Math.sqrt(((n_core*n_core)/(n_clad*n_clad))*(Math.sin(theta)*Math.sin(theta))-1))
        - 2*m*Math.PI);
        return Ecuacion;
    }
    public static double resultado_FuncionTM(double theta, double n_core, double n_clad, double long_Onda, double altura_fibra ,double m){
        /*Esta función recibe los argumentos necesarios para calcular la suma de todos los cambios de fase en ondas TM */
        double Ecuacion = (double)(((4 * Math.PI * n_core*altura_fibra*Math.cos(theta))/(long_Onda)) 
        - 4*Math.atan(((n_core/(n_clad*Math.cos(theta))))*Math.sqrt(((n_core*n_core)/(n_clad*n_clad))*(Math.sin(theta)*Math.sin(theta))-1))
        - 2*m*Math.PI);
        return Ecuacion;
    }
    public static void main(String[] args) {
        List <Double> raices = new ArrayList<>(); //Lista vacía para almacenar las raíces encontradas 
        int tamaño_AnteriorLista = raices.size(); //Variable para almacenar el tamaño de la lista
        int tamaño_ActualLista; //Variable para almacenar el tamaño de la lista
        double incremento = 0.01; //Incremento para el ángulo theta
        double n_core = 1.5; //Índice de refracción del núcleo
        double n_clad = 1.0; //Índice de refracción del revestimiento
        double long_onda = 1E-6; //Longitud de onda
        double altura_fibra = 1E-6; //Altura de la fibra
        double m=0; //Orden de la interferencia
        boolean condition = true; //Variable de control para el bucle while
        String tipo_onda = "TE"; //Tipo de onda (TE o TM)
        double theta_critico = (double)(Math.asin(n_clad/n_core)); //Ángulo crítico para iniciar con el cálculo numérico
        double theta_derecha = theta_critico+incremento; //Límite derecho del intervalo
        double theta_izquierda = theta_critico; //Límite izquierdo del intervalo, se inicia desde el ángulo crítico
        if(tipo_onda.equalsIgnoreCase("TE")){
            //Se evalua la función en el ángulo crítico para iniciar a trabajar desde reflexión total interna
            while (condition) {
            double resultado_izquierda = resultado_FuncionTE(theta_izquierda, n_core, n_clad, long_onda, altura_fibra, m);   //Se evalua la función con el valor menor
            double resultado_derecha = resultado_FuncionTE(theta_derecha, n_core, n_clad, long_onda, altura_fibra, m);       //Se evalua la función con el valor mayor
            if (resultado_derecha*resultado_izquierda < 0 && Math.abs(resultado_derecha+resultado_izquierda) < 1){
                //Se encuentra un cambio de signo, por lo tanto hay una raíz entre estos valores
                double raiz = (theta_derecha+theta_izquierda)/2; //Se propone que esta es la raíz
                double resultado_raiz = resultado_FuncionTE(raiz, n_core, n_clad, long_onda, altura_fibra, m); //Se evalua la función en la raíz propuesta
                if(resultado_izquierda*resultado_raiz < 0){
                    //Significa que la raíz está entre el límite izquierdo y la raíz propuesta
                    theta_derecha = raiz; //Se ajusta el límite derecho y se repite desde el paso 2
                }else if (resultado_izquierda*resultado_raiz > 0) {
                    //Significa que la raíz está entre el límite derecho y la raíz propuesta
                    theta_izquierda = raiz; //Se ajusta el límite izquierdo y se repite desde el paso 2
                }
                if(resultado_raiz*resultado_izquierda < 1E-5){
                    //Se encontró la raíz exacta, entonces la variable raíz es la solución 
                    raices.add(raiz); //Se agrega la raíz a la lista
                    raices.add(m); //Se agrega el orden de la interferencia a la lista
                    theta_izquierda =raiz+incremento; //Se ajusta el límite izquierdo
                    theta_derecha = theta_izquierda + incremento; //Se ajusta el límite derecho  
                }
            }else{
                //No se encontró un cambio de signo, por lo tanto se ajustan los límites, se incrementa el ángulo y se repite el proceso
                theta_izquierda += incremento; //Se incrementa el límite izquierdo
                theta_derecha += incremento; //Se incrementa el límite derecho
            }

            if (theta_derecha >= Math.PI/2){
                //Se ha llegado al límite superior del intervalo, por lo tanto se detiene el bucle
                tamaño_ActualLista = raices.size(); //Se obtiene el tamaño actual de la lista para comparar
                if (tamaño_ActualLista == tamaño_AnteriorLista){
                    //No se encontró ninguna raíz, por lo tanto se detiene el bucle pues ya no hay sentido en evaluar más ordenes
                    for (int i = 0; i < raices.size(); i+=2) {
                        System.out.println("Raíz " + i/2 + ": " + raices.get(i) + " m = " + raices.get(i+1));
                    }
                    condition = false; //Se cambia la variable de control para salir del bucle
                }
                m++; //Se incrementa el orden de la interferencia
                theta_izquierda = theta_critico; //Se reinicia el límite izquierdo
                theta_derecha = theta_izquierda + incremento; //Se reinicia el límite derecho
                tamaño_AnteriorLista = raices.size(); //Se actualiza el tamaño anterior de la lista
                }
            }
        }
        else if(tipo_onda.equalsIgnoreCase("TM")){
            //Se evalua la función en el ángulo crítico para iniciar a trabajar desde reflexión total interna
            while (condition) {
            double resultado_izquierda = resultado_FuncionTM(theta_izquierda, n_core, n_clad, long_onda, altura_fibra, m);   //Se evalua la función con el valor menor
            double resultado_derecha = resultado_FuncionTM(theta_derecha, n_core, n_clad, long_onda, altura_fibra, m);       //Se evalua la función con el valor mayor
            if (resultado_derecha*resultado_izquierda < 0 && Math.abs(resultado_derecha+resultado_izquierda) < 1){
                //Se encuentra un cambio de signo, por lo tanto hay una raíz entre estos valores
                double raiz = (theta_derecha+theta_izquierda)/2; //Se propone que esta es la raíz
                double resultado_raiz = resultado_FuncionTM(raiz, n_core, n_clad, long_onda, altura_fibra, m); //Se evalua la función en la raíz propuesta
                if(resultado_izquierda*resultado_raiz < 0){
                    //Significa que la raíz está entre el límite izquierdo y la raíz propuesta
                    theta_derecha = raiz; //Se ajusta el límite derecho y se repite desde el paso 2
                }else if (resultado_izquierda*resultado_raiz > 0) {
                    //Significa que la raíz está entre el límite derecho y la raíz propuesta
                    theta_izquierda = raiz; //Se ajusta el límite izquierdo y se repite desde el paso 2
                }
                if(resultado_raiz*resultado_izquierda < 1E-5){
                    //Se encontró la raíz exacta, entonces la variable raíz es la solución 
                    raices.add(raiz); //Se agrega la raíz a la lista
                    raices.add(m); //Se agrega el orden de la interferencia a la lista
                    theta_izquierda =raiz+incremento; //Se ajusta el límite izquierdo
                    theta_derecha = theta_izquierda + incremento; //Se ajusta el límite derecho  
                }
            }else{
                //No se encontró un cambio de signo, por lo tanto se ajustan los límites, se incrementa el ángulo y se repite el proceso
                theta_izquierda += incremento; //Se incrementa el límite izquierdo
                theta_derecha += incremento; //Se incrementa el límite derecho
            }

            if (theta_derecha >= Math.PI/2){
                //Se ha llegado al límite superior del intervalo, por lo tanto se detiene el bucle
                tamaño_ActualLista = raices.size(); //Se obtiene el tamaño actual de la lista para comparar
                if (tamaño_ActualLista == tamaño_AnteriorLista){
                    //No se encontró ninguna raíz, por lo tanto se detiene el bucle pues ya no hay sentido en evaluar más ordenes
                    for (int i = 0; i < raices.size(); i+=2) {
                        System.out.println("Raíz " + i/2 + ": " + raices.get(i) + " m = " + raices.get(i+1));
                    }
                    condition = false; //Se cambia la variable de control para salir del bucle
                }
                m++; //Se incrementa el orden de la interferencia
                theta_izquierda = theta_critico; //Se reinicia el límite izquierdo
                theta_derecha = theta_izquierda + incremento; //Se reinicia el límite derecho
                tamaño_AnteriorLista = raices.size(); //Se actualiza el tamaño anterior de la lista
                }
            }
        }
    }
}