package Taller_1;
import java.util.ArrayList;
import java.util.List;

public class spaghetti {
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

    public static double resultado_FuncionModosTEPares(double phi){
        double n_clad = 1.0 ; //Índice de refracción del revestimiento
        double n_core = 1.5; //Índice de refracción del núcleo
        double long_Onda = 1E-6; //Longitud de onda
        double altura_fibra = 1E-6; //Altura de la fibra
        double Ecuacion = Math.pow(phi,2)+Math.pow(phi*Math.tan(phi),2)
        -Math.pow((2*Math.PI*altura_fibra)/(long_Onda*2),2)*(Math.pow(n_core, 2)-Math.pow(n_clad, 2));
        /*Con esta ecuación es posible conocer los phi, con lo que es posible conocer los phi.
         * Esto ayuda a hallar los valores de kappa y gamma para finalmente obtener el beta.
         */
        return Ecuacion;
    }
    public static double resultado_FuncionModosTEImpares(double phi){
        double n_clad = 1.0 ; //Índice de refracción del revestimiento
        double n_core = 1.5; //Índice de refracción del núcleo
        double long_Onda = 1E-6; //Longitud de onda
        double altura_fibra = 1E-6; //Altura de la fibra
        double Ecuacion = Math.pow(phi,2)+Math.pow(phi/Math.tan(phi),2)
        -Math.pow((2*Math.PI*altura_fibra)/(long_Onda*2),2)*(Math.pow(n_core, 2)-Math.pow(n_clad, 2));;
        return Ecuacion;
    }
    public static double resultado_FuncionModosTMPares(double phi){
        double n_clad = 1.0 ; //Índice de refracción del revestimiento
        double n_core = 1.5; //Índice de refracción del núcleo
        double long_Onda = 1E-6; //Longitud de onda
        double altura_fibra = 1E-6; //Altura de la fibra
        double Ecuacion = Math.pow(phi,2)+Math.pow(Math.pow(n_clad/n_core,2)*phi*Math.tan(phi),2)
        -Math.pow((2*Math.PI*altura_fibra)/(long_Onda*2),2)*(Math.pow(n_core, 2)-Math.pow(n_clad, 2));
        return Ecuacion;
    }
    public static double resultado_FuncionModosTMImpares(double phi){
        double n_clad = 1.0 ; //Índice de refracción del revestimiento
        double n_core = 1.5; //Índice de refracción del núcleo
        double long_Onda = 1E-6; //Longitud de onda
        double altura_fibra = 1E-6; //Altura de la fibra
        double Ecuacion = Math.pow(phi,2)+Math.pow(Math.pow(n_clad/n_core,2)*phi/Math.tan(phi),2)
        -Math.pow((2*Math.PI*altura_fibra)/(long_Onda*2),2)*(Math.pow(n_core, 2)-Math.pow(n_clad, 2));
        return Ecuacion;
    }

    public static List<Double> pasos_3_5MetodoOndulatorio(List<Double> raices_phi, List<Double> raices_alfa){
        double altura_fibra = 1E-6; //Altura de la fibra
        double n_core = 1.5;  //Índice de refracción del núcleo
        double n_clad = 1.0; //Índice de refracción del revestimiento
        double long_Onda = 1E-6; //Longitud de onda
        double numero_onda = (2*Math.PI)/long_Onda; //Número de onda
        List <Double> raices_kappa = new ArrayList<>();
        List <Double> raices_gamma = new ArrayList<>();
        List <Double> betas_pemitidos = new ArrayList<>();
        List <Double> thetas_permitidos = new ArrayList<>();
        for (int i = 0; i < raices_phi.size(); i++){
            /*En este ciclo se calculan los valores de kappa y gamma */
            double phi = raices_phi.get(i); //Se obtiene el phi
            double alfa = raices_alfa.get(i); //Se obtiene el alfa
            double kappa = phi * 2/altura_fibra;
            double gamma = alfa * 2/altura_fibra;
            raices_gamma.add(gamma); //Se agrega el gamma a la lista
            raices_kappa.add(kappa); //Se agrega el kappa a la lista
        }
        for (int i = 0; i < raices_phi.size(); i++){
            /*En este ciclo se calculan los valores de beta, a través del promedio con kappa y gamma para disminuir el error */
            double kappa = raices_kappa.get(i); //Se obtiene el kappa
            double gamma = raices_gamma.get(i); //Se obtiene el gamma
            double beta_kappa = Math.sqrt(-Math.pow(kappa,2)+Math.pow((n_core*numero_onda),2)); //Se obtiene el beta_kappa
            double beta_gamma = Math.sqrt(Math.pow(gamma,2)+Math.pow((n_clad*numero_onda),2)); //Se obtiene el beta_gamma
            double beta_promedio = (beta_gamma+beta_kappa)/2 ;
            betas_pemitidos.add(beta_promedio);
        }
        for (int i = 0; i < raices_phi.size(); i++){
            /* En este ciclo se hallan los valores permitidos para el ángulo de incidencia theta */
            double beta = betas_pemitidos.get(i);
            double theta = Math.asin((beta)/(n_core*numero_onda));
            thetas_permitidos.add(theta);
        }
        System.out.println("Raices de beta: " + betas_pemitidos);
        return thetas_permitidos;
    }

    public static void main(String[] args) {
        double n_clad = 1.0; //Índice de refracción del revestimiento
        double n_core = 1.5; //Índice de refracción del núcleo 
        double long_Onda = 1E-6; //Longitud de onda
        double altura_fibra = 1E-6; //Altura de la fibra
        double numero_onda = (2*Math.PI)/long_Onda; //Número de onda    
        double fin = (numero_onda*altura_fibra)/2*Math.sqrt(Math.pow(n_core,2)-Math.pow(n_clad,2)); //Límite superior
        double inicio = 0; //Límite inferior
        // Modos Transversales Electricos Pares
        List <Double> raices_phi = taller1_MetodoBiseccion.metodo_Biseccion(taller1_ModosOndulatorios::resultado_FuncionModosTEPares, inicio, fin);
        List <Double> raices_alfa = new ArrayList<>();
        List <Double> n_eff = new ArrayList<>(); //Lista de índices de refracción efectivos

        for (int i = 0; i < raices_phi.size(); i++){
            /*En este ciclo se calulan los valores de alfa */
            double phi = raices_phi.get(i); //Se obtiene el phi
            double alfa = phi * Math.tan(phi); //Se obtiene el alfa
            raices_alfa.add(alfa); //Se agrega el alfa a la lista
        }
        List <Double> thetas_permitidos = taller1_Pasos3_5MetodoOndulatorio.pasos_3_5MetodoOndulatorio(raices_phi, raices_alfa);
        for (int i=0 ; i<raices_phi.size(); i++){
            double aux = n_core*Math.sin(thetas_permitidos.get(i)); //Se obtiene el índice de refracción efectivo
            n_eff.add(aux);; 
        }
        for(int i=0 ; i<raices_phi.size(); i++){
            thetas_permitidos.set(i, Math.toDegrees(thetas_permitidos.get(i))); //Convierte las raíces a grados
        }
        System.out.println("Thetas Permitidos TE Pares: " + thetas_permitidos);
        System.out.println("n_eff para el modo TE Pares: " + n_eff); //Se imprime la lista de índices de refracción efectivos
        n_eff.clear(); //Limpiamos la lista de índices de refracción efectivos
        raices_alfa.clear(); //Limpiamos la lista de raices_alfa

        // Modos Transversales Electricos Impares
        raices_phi = taller1_MetodoBiseccion.metodo_Biseccion(taller1_ModosOndulatorios::resultado_FuncionModosTEImpares, inicio, fin);
        for (int i = 0; i < raices_phi.size(); i++){
            /*En este ciclo se calulan los valores de alfa */
            double phi = raices_phi.get(i); //Se obtiene el phi
            double alfa = -phi / Math.tan(phi); //Se obtiene el alfa
            raices_alfa.add(alfa); //Se agrega el alfa a la lista
        }
        thetas_permitidos = taller1_Pasos3_5MetodoOndulatorio.pasos_3_5MetodoOndulatorio(raices_phi, raices_alfa);
        for (int i=0 ; i<raices_phi.size(); i++){
            double aux = n_core*Math.sin(thetas_permitidos.get(i)); //Se obtiene el índice de refracción efectivo
            n_eff.add(aux);; 
        }
        for(int i=0 ; i<raices_phi.size(); i++){
            thetas_permitidos.set(i, Math.toDegrees(thetas_permitidos.get(i))); //Convierte las raíces a grados
        }
        System.out.println("Thetas Permitidos TE Impares: " + thetas_permitidos);
        System.out.println("n_eff para el modo TE Impares: " + n_eff); //Se imprime la lista de índices de refracción efectivos
        n_eff.clear(); //Limpiamos la lista de índices de refracción efectivos
        raices_alfa.clear(); //Limpiamos la lista de raices_alfa

        // Modos Transversales Magneticos Pares
        raices_phi = taller1_MetodoBiseccion.metodo_Biseccion(taller1_ModosOndulatorios::resultado_FuncionModosTMPares, inicio, fin);
        for (int i = 0; i < raices_phi.size(); i++){
            /*En este ciclo se calulan los valores de alfa */
            double phi = raices_phi.get(i); //Se obtiene el phi
            double alfa = Math.pow(n_clad/n_core,2)*phi * Math.tan(phi); //Se obtiene el alfa
            raices_alfa.add(alfa); //Se agrega el alfa a la lista
        }
        thetas_permitidos = taller1_Pasos3_5MetodoOndulatorio.pasos_3_5MetodoOndulatorio(raices_phi, raices_alfa);
        for (int i=0 ; i<raices_phi.size(); i++){
            double aux = n_core*Math.sin(thetas_permitidos.get(i)); //Se obtiene el índice de refracción efectivo
            n_eff.add(aux);; 
        }
        for(int i=0 ; i<raices_phi.size(); i++){
            thetas_permitidos.set(i, Math.toDegrees(thetas_permitidos.get(i))); //Convierte las raíces a grados
        }
        System.out.println("Thetas Permitidos TM Pares: " + thetas_permitidos);
        System.out.println("n_eff para el modo TM Pares: " + n_eff); //Se imprime la lista de índices de refracción efectivos
        n_eff.clear(); //Limpiamos la lista de índices de refracción efectivos
        raices_alfa.clear(); //Limpiamos la lista de raices_alfa

        //Modos Transversales Magneticos Impares
        raices_phi = taller1_MetodoBiseccion.metodo_Biseccion(taller1_ModosOndulatorios::resultado_FuncionModosTMImpares, inicio, fin); 
        for (int i = 0; i < raices_phi.size(); i++){
            /*En este ciclo se calulan los valores de alfa */
            double phi = raices_phi.get(i); //Se obtiene el phi
            double alfa = -Math.pow(n_clad/n_core,2)*phi / Math.tan(phi); //Se obtiene el alfa
            raices_alfa.add(alfa); //Se agrega el alfa a la lista
        }
        thetas_permitidos = taller1_Pasos3_5MetodoOndulatorio.pasos_3_5MetodoOndulatorio(raices_phi, raices_alfa);
        for (int i=0 ; i<raices_phi.size(); i++){
            double aux = n_core*Math.sin(thetas_permitidos.get(i)); //Se obtiene el índice de refracción efectivo
            n_eff.add(aux);; 
        }
        for(int i=0 ; i<raices_phi.size(); i++){
            thetas_permitidos.set(i, Math.toDegrees(thetas_permitidos.get(i))); //Convierte las raíces a grados
        }
        System.out.println("Thetas Permitidos TM Impares: " + thetas_permitidos);
        System.out.println("n_eff para el modo TM Impares: " + n_eff); //Se imprime la lista de índices de refracción efectivos
    }
}
