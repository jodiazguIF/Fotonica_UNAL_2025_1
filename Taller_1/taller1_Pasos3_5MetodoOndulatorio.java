package Taller_1;
import java.util.ArrayList;
import java.util.List;

public class taller1_Pasos3_5MetodoOndulatorio {
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
        return thetas_permitidos;
    }
}
