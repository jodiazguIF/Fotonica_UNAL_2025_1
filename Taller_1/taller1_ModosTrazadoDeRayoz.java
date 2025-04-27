package Taller_1;

public class taller1_ModosTrazadoDeRayoz {
    public static double resultado_FuncionTE(double theta, double orden_Modo){
        /*Esta función recibe los argumentos necesarios para calcular la suma de todos los cambios de fase en ondas TE */
        double n_core = 1.5; //Índice de refracción del núcleo
        double n_clad = 1.0; //Índice de refracción del revestimiento
        double long_Onda = 1E-6; //Longitud de onda
        double altura_fibra = 1E-6; //Altura de la fibra
        double Ecuacion = (double)(((4 * Math.PI * n_core*altura_fibra*Math.cos(theta))/(long_Onda)) 
        - 4*Math.atan(((n_clad/(n_core*Math.cos(theta))))*Math.sqrt(((n_core*n_core)/(n_clad*n_clad))*(Math.sin(theta)*Math.sin(theta))-1))
        - 2*orden_Modo*Math.PI);
        return Ecuacion;
    }
    public static double resultado_FuncionTM(double theta, double orden_Modo){
        /*Esta función recibe los argumentos necesarios para calcular la suma de todos los cambios de fase en ondas TM */
        double n_core = 1.5; //Índice de refracción del núcleo
        double n_clad = 1.0; //Índice de refracción del revestimiento
        double long_Onda = 1E-6; //Longitud de onda
        double altura_fibra = 1E-6; //Altura de la fibra
        double Ecuacion = (double)(((4 * Math.PI * n_core*altura_fibra*Math.cos(theta))/(long_Onda)) 
        - 4*Math.atan(((n_core/(n_clad*Math.cos(theta))))*Math.sqrt(((n_core*n_core)/(n_clad*n_clad))*(Math.sin(theta)*Math.sin(theta))-1))
        - 2*orden_Modo*Math.PI);
        return Ecuacion;
    }
}