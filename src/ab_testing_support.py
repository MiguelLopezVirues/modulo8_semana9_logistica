
import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns

from statsmodels.stats.multicomp import pairwise_tukeyhsd
from statsmodels.stats.proportion import proportions_ztest # para hacer el ztest
import scikit_posthocs as sp


from src import data_visualization_support as dvs
import pingouin as pg



## ASUMPTIONS


# Normality
class AB_testing():
    def __init__(self, df, columna_metrica, alpha, verbose=False, dependiente=False, proporcion=False, resample=False):
        self.df = df.copy()
        self.columna_metrica = columna_metrica
        # self.df[self.columna_metrica] = ((self.df[self.columna_metrica] - self.df[self.columna_metrica].min()) 
        #                     / (self.df[self.columna_metrica].max() - self.df[self.columna_metrica].min()))

        self.alpha = alpha
        self.verbose = verbose
        self.parametrico = False
        self.diferencias = False
        self.dependiente = dependiente
        self.proporcion = proporcion
        self.small_sample = False
        self.resample = resample


    def evaluar_ab_testing(self, columna_grupo):
        # plotear distribuciones
        if self.verbose:
            print(f"Evaluando la distribución de la variable {self.columna_metrica} mediante histograma...")
            dvs.graficar_distribuciones(self.df, columna_grupo, self.columna_metrica)

        muestras_iguales = self.evaluar_tamanio_muestras(columna_grupo)

        if not muestras_iguales and self.verbose:
            print("Tamaños muestrales diferentes entre grupos. Distribuciones actualizadas para muestras igualadas:.")
            dvs.graficar_distribuciones(self.df, columna_grupo, self.columna_metrica)

        # EVALUAR SUPUESTOS Y APLICAR TEST PARAMÉTRICO
        if self.tests_normalidad(columna_grupo) and self.test_homogeneidad(columna_grupo):
            # if self.verbose:
            print("\nSe cumplen los supuestos de normalidad y homocedasticidad. Procediendo con tests paramétricos:")

            self.parametrico = True
            self.test_parametrico(columna_grupo)

        # O NO PARAMÉTRICO
        else:
            # if self.verbose:
            print("\nLos supuestos paramétricos no se cumplen. Procediendo con tests NO paramétricos.")
            print("----------------------------------------------------------------------------------\n")

            self.test_no_parametrico(columna_grupo)
            
        # APLICAR ANÁLISIS POST-HOC
        if self.diferencias:
            self.post_hoc(columna_grupo)


    def separar_grupos(self,columna_grupo, sample_size=None):
        if isinstance(sample_size,int):
            return [self.df.loc[self.df[columna_grupo]==grupo,self.columna_metrica].sample(sample_size) for grupo in self.df[columna_grupo].unique()]
        return [self.df.loc[self.df[columna_grupo]==grupo,self.columna_metrica] for grupo in self.df[columna_grupo].unique()]

    def evaluar_tamanio_muestras(self, columna_grupo):

        # comprobar condicion de que todos los grupos tienen mismo tamaño muestral
        tamanio_muestral_grupos = self.df.groupby(columna_grupo, observed=False)[self.columna_metrica].count()

        tamanio_min = tamanio_muestral_grupos.min()
        if tamanio_min <= 100:
            print("Las muestras de los siguientes grupos son pequeñas:")
            display(tamanio_muestral_grupos[tamanio_muestral_grupos < 100])
            self.small_sample = True

        # si el tamaño muestral es 0, elevar ValueError
        if tamanio_min == 0:
            raise ValueError(f"One or more groups in '{columna_grupo}' have no data.")
        
        # evaluar muestras igual tamaño
        muestras_igual_tamanio = all(tamanio_muestral_grupos == tamanio_min)

        # si no, genero y devuelvo df con remuestro
        if not muestras_igual_tamanio and self.resample:
            if self.verbose:
                print(f"\n\nMuestras de tamaño desigual detectadas. Aplicando remuestreo para igualar a la muestra más pequeña de N = {tamanio_min}")
                print("\n\n························································································")
            
            # generar df remuestreo 
            df_remuestro_total = pd.DataFrame()
            for grupo in self.df[columna_grupo].unique():
                df_grupo_remuestreo = self.df.loc[self.df[columna_grupo]==grupo,[columna_grupo,self.columna_metrica]].sample(tamanio_min)
                df_remuestro_total = pd.concat([df_remuestro_total,df_grupo_remuestreo])

            # mostrar nuevo reparto
            if self.verbose:
                print("Conjunto de datos con remuestreo:")
                display(df_remuestro_total.groupby(columna_grupo, observed=False)[[self.columna_metrica]].count())
            self.df_grupos = df_remuestro_total
        
        # si tamaños iguales, no hay cambio
        elif muestras_igual_tamanio and self.verbose:
            print("Tamaño muestral uniforme entre los grupos.")


    def tests_normalidad(self, columna_grupo):
        print(f"Aplicando tests de normalidad.\n".upper()) 
        resultados = []
        for grupo in self.df[columna_grupo].unique():
            grupo_df = self.df[self.df[columna_grupo] == grupo]
            media = np.mean(grupo_df[self.columna_metrica])
            desv_tip = np.std(grupo_df[self.columna_metrica], axis=0)

            # realizar tests. Primero Kolmogorov-Smirnoff, luego Shapiro-Wilk
            p_value_k = np.nan
            resultado_k = np.nan
            if grupo_df.shape[0] > 100:
                _, p_value_k = stats.kstest(grupo_df[self.columna_metrica],'norm', args=(media, desv_tip))
                resultado_k = "True" if p_value_k > self.alpha else "False"
            elif self.verbose:
                print("Tamaño muestral menor a 100, no se realiza test Kolmogorov-Smirnoff.")

            if grupo_df.shape[0] > 5000:
                grupo_df = grupo_df.sample(5000)
            _, p_value_s = stats.shapiro(grupo_df[self.columna_metrica])
            resultado_s = "True" if p_value_s > self.alpha else "False"

            # Agregar resultados a la lista
            resultados.append({
                'Grupo': grupo,
                'KS p-valor': round(p_value_k, 4),
                'Normalidad KS': resultado_k,
                'SW p-valor': round(p_value_s, 4),
                'Normalidad SW': resultado_s
            })

        resultados_df = pd.DataFrame(resultados,columns=resultados[0].keys())

        print("Definición de las hipótesis:")
        print("H0: La variable medida sigue una distribución normal en el grupo evaluado.")
        print("H1: La variable medida NO sigue una distribución normal en el grupo evaluado.")
        
        if self.verbose:
            # Generar siempre el informe en estilo formateado
            print("     Normality Tests by Group          ")
            display(resultados_df)

        # print("===============================================================")
        # print("     Grupo      KS p-valor  Normalidad KS   SW p-valor  Normalidad SW")
        # print("---------------------------------------------------------------")

        # for row in resultados:
        #     print(f"{row['Grupo']:>10} {row['KS p-valor']:>12} {row['Normalidad KS']:>14} {row['SW p-valor']:>13} {row['Normalidad SW']:>14}")
        #     print("---------------------------------------------------------------")

        conclusiones = {
            "condicion": {
            "Normalidad total": lambda: all(res['Normalidad KS'] == "True" and res['Normalidad SW'] == "True" for res in resultados),
            "Soft_fail": lambda: all(res['Normalidad KS'] == "True" or res['Normalidad SW'] == "True" for res in resultados),
            "Hard_fail": lambda: any(res['Normalidad KS'] == "False" and res['Normalidad SW'] == "False" for res in resultados)
            },
            "mensaje": {
            "Normalidad total": "Todos los grupos cumplen el supuesto de normalidad según Kolmogorov-Smirnoff y Shapiro-Wilk.",
            "Soft_fail": "Existe discordancia entre Kolmogorov-Smirnoff y Shapiro-Wilk para alguno de los grupos. Realizar conclusiones con precaución de aquí en adelante.",
            "Hard_fail": "Al menos uno de los grupos no cumple el supuesto de normalidad."
            }
        }

        if self.verbose:
            for conclusion, condicion in conclusiones["condicion"].items():
                if condicion():
                    print(conclusiones["mensaje"][conclusion])

    
        return all(res['Normalidad KS'] == "True" or res['Normalidad SW'] == "True" for res in resultados)


    # Homoscedasticity
    def test_homogeneidad (self, columna_grupo):
        
        """
        Evalúa la homogeneidad de las varianzas entre grupos para una métrica específica en un DataFrame dado.

        Params:
        - columna (str): El nombre de la columna que se utilizará para dividir los datos en grupos.
        - columna_grupo (str): El nombre de la columna que se utilizará para evaluar la homogeneidad de las varianzas.

        Returns:
        No Returns nada directamente, pero imprime en la consola si las varianzas son homogéneas o no entre los grupos.
        Se utiliza la prueba de Levene para evaluar la homogeneidad de las varianzas. Si el valor p resultante es mayor que 0.05,
        se concluye que las varianzas son homogéneas; de lo contrario, se concluye que las varianzas no son homogéneas.
        """
        
        # if self.verbose:
        print(f"Aplicando el test de homogeneidad.\n".upper()) 
        print("Definición de las hipótesis:")
        print("H0: Los grupos presentan varianzas equivalentes u homogéneas.")
        print("H1: Los grupos NO presentan varianzas equivalentes u homogéneas.\n")
        _, p_value = stats.levene(*self.separar_grupos(columna_grupo))
        if p_value > self.alpha:
            mensaje = f"Con un p-valor igual a {p_value} < alpha {self.alpha}, en la variable {columna_grupo} las varianzas son homogéneas entre grupos."
            return True
        else:
            mensaje = f"Con un p-valor igual a {p_value} > alpha {self.alpha}, en la variable {columna_grupo} las varianzas NO son homogéneas entre grupos."
        
        if self.verbose:
            print(mensaje)

    def test_parametrico(self, columna_grupo):
        groups_df_list = self.separar_grupos(columna_grupo)
        tests = {
            len(groups_df_list) == 2: [
                "Welch’s t-test",
                lambda group_dfs: stats.ttest_ind(*group_dfs, equal_var=False)
            ],
            len(groups_df_list) == 2 & self.dependiente: [
                "T-test for dependent samples",
                lambda group_dfs: stats.ttest_rel(*group_dfs)
            ],
            len(groups_df_list) > 2: [
                "Welch ANOVA",
                lambda group_dfs: pg.welch_anova(
                    dv=self.columna_metrica,
                    between=columna_grupo,
                    data=self.da,
                )
            ],
            self.proporcion: [
                "Z-Test for proportions",
                lambda _: self.z_test()
            ],

        }

        nombre_test, p_value = [(test[0], test[1](groups_df_list)[1]) for condition,test in tests.items() if condition][0]

        if p_value > self.alpha:
            mensaje = f"Con un p-valor igual a {round(p_value,3)} > alpha {self.alpha}, en la variable {columna_grupo} no existen diferencias entre la medias de los grupos."
        else:
            mensaje = f"Con un p-valor igual a {round(p_value,3)} > alpha {self.alpha}, en la variable {columna_grupo}, las medias muestran distribuciones diferentes."
            self.diferencias = True
        
        # if self.verbose:
        print(f"Aplicando el test de {nombre_test}.\n".upper())  
        print("Definición de las hipótesis:")
        print("H0: Los grupos presentan medianas equivalentes u homogéneas.")
        print("H1: Los grupos NO presentan medianas equivalentes u homogéneas.\n")
        
        print(mensaje)
        print("----------------------------------------------------------------------------------\n")

    def test_no_parametrico(self, columna_grupo, dependiente=False, proporcion=False):
        groups_df_list = self.separar_grupos(columna_grupo)
        tests = {
            len(groups_df_list) == 2: [
                "Mann-Whitney U",
                lambda group_dfs: stats.mannwhitneyu(*group_dfs, alternative='two-sided')
            ],
            len(groups_df_list) == 2 & self.dependiente: [
                "Wilcoxon Signed-Rank Test",
                lambda group_dfs: stats.wilcoxon(*group_dfs)
            ],
            len(groups_df_list) > 2: [
                "Kruskal-Wallis Test",
                lambda group_dfs: stats.kruskal(*group_dfs)
            ],
            self.proporcion: [
                "Chi-Square Test",
                lambda group_dfs: stats.chi2_contingency(group_dfs)
            ],
            self.proporcion & self.small_sample: [
                "Fisher’s Exact Test",
                lambda group_dfs: stats.fisher_exact(group_dfs)
            ]

        }

        nombre_test, p_value = [(test[0], test[1](groups_df_list)[1]) for condition,test in tests.items() if condition][0]

        if p_value > self.alpha:
            mensaje = f"Con un p-valor igual a {round(p_value,3)} > alpha {self.alpha}, en la variable {columna_grupo} no existen diferencias entre la medianas de los grupos."
        else:
            mensaje = f"Con un p-valor igual a {round(p_value,3)} > alpha {self.alpha}, en la variable {columna_grupo}, las medianas muestran distribuciones diferentes."
            self.diferencias = True
        
        # if self.verbose:
        print(f"Aplicando el test de {nombre_test}:\n".upper())  
        print("Definición de las hipótesis:")
        print("H0: Los grupos presentan medianas equivalentes u homogéneas.")
        print("H1: Los grupos NO presentan medianas equivalentes u homogéneas.")
        
        print(f"\n{mensaje}")
        print("----------------------------------------------------------------------------------\n")

    def plotear_repartos(self, columna_grupo):
        if self.parametrico:
            estimator = "mean"
            estimator_sp = "medias"
        else:
            estimator = "median"
            estimator_sp = "medianas"
        
        plt.figure(figsize=(15,8))
        plt.title(f"Diferencia de {estimator_sp} para los grupos de {columna_grupo}.")

        ax = sns.barplot(data=self.df,
                    x=columna_grupo,
                    y=self.columna_metrica,
                    hue=columna_grupo,
                    estimator=estimator,
                    palette="mako")
        
        ax.tick_params("x",labelrotation=45)
        
        dvs.plot_bar_labels(ax=ax)
        plt.show()



    def comprobar_pvalue(self, pvalor, alpha=0.05):
        """
        Comprueba si el valor p es significativo.

        Params:
            - pvalor: Valor p obtenido de la prueba estadística.
            - alpha (opcional): Nivel de significancia. Por defecto es 0.05.

        Returns:
            No devuelve nada.
        """
        if pvalor < alpha:
            print(f"El p-valor de la prueba es {round(pvalor, 2)}, por lo tanto, hay diferencias significativas entre los grupos.")
        else:
            print(f"El p-valor de la prueba es {round(pvalor, 2)}, por lo tanto, no hay evidencia de diferencias significativas entre los grupos.")


    def z_test(self, grupos_dfs):
        """
        Realiza el test Z para proporciones.

        Calcula el valor Z y el p-valor de la prueba y lo imprime en la consola.

        Params: 
            No recibe nigún parámetros

        Returns:
            No devuelve nada.
        """
        control, test = grupos_dfs

        # calculamos el número de usuarios que han convertido en cada uno de los tratamientos y creamos una lista (esto es asi porque el método de python para hacer el ztest nos pide una lista)
        convertidos = [control.sum(), test.sum()]
        # contamos el número de filas que tenemos para cada uno de los grupos, es decir, calculamos el tamaño muestral del grupo control y grupo test. Todo esto lo almacenamos en una lista igual que antes. 
        tamaños_muestrales = [control.count(), test.count()]

        resultados_test = proportions_ztest(convertidos, tamaños_muestrales)

        return resultados_test
        # print(f"El estadístico de prueba (Z) es: {round(resultados_test[0], 2)}, el p-valor es {round(resultados_test[1], 2)}")
        
        # # Interpretar los resultados
        # self.comprobar_pvalue(resultados_test[1])



    def test_anova(self, grupos_dfs):
        """
        Realiza el test ANOVA para comparar las medias de múltiples grupos.

        Calcula el estadístico F y el valor p de la prueba y lo imprime en la consola.
        
        Params: 
            No recibe nigún parámetros

        Returns:
            No devuelve nada.
        """
        categorias = grupos_dfs
        statistic, p_value = stats.f_oneway(*categorias)

        return p_value
        # print("Estadístico F:", statistic)
        # print("Valor p:", p_value)

        # self.comprobar_pvalue(p_value)

    def test_t(self, grupos_dfs):
        """
        Realiza el test t de Student para comparar las medias de dos grupos independientes.

        Calcula el estadístico t y el valor p de la prueba y lo imprime en la consola.

        Params: 
            No recibe nigún parámetros

        Returns:
            No devuelve nada.
        """
        t_stat, p_value = stats.ttest_ind(*grupos_dfs)

        return p_value

        # print("Estadístico t:", t_stat)
        # print("Valor p:", p_value)

        # self.comprobar_pvalue(p_value)

    def test_t_dependiente(self,columna_grupo):
        """
        Realiza el test t de Student para comparar las medias de dos grupos dependientes.

        Calcula el estadístico t y el valor p de la prueba y lo imprime en la consola.

        Params: 
            No recibe nigún parámetros

        Returns:
            No devuelve nada.
        """
        categorias = self.separar_grupos(columna_grupo)

        t_stat, p_value = stats.ttest_rel(*categorias)

        return p_value

        # print("Estadístico t:", t_stat)
        # print("Valor p:", p_value)

        # self.comprobar_pvalue(p_value)



    def post_hoc(self, columna_grupo):
        if self.parametrico:
            # if self.verbose:

            self.test_tukey(columna_grupo)

        else:
            self.test_dunn(columna_grupo)
        
        self.plotear_repartos(columna_grupo)

    def test_tukey(self, columna_grupo):
        print(f"Aplicando el test de Tukey, post-hoc no paramétrico\n".upper()) 
        print("Test de Tukey (comparaciones por pares):")
        print("Hipótesis Nula (H0): Las medias de los grupos comparados son iguales.")
        print("Hipótesis Alternativa (H1): Las medias de los grupos comparados son significativamente diferentes.")

        tukey_results = pairwise_tukeyhsd(endog=self.df[self.columna_metrica],groups=self.df[columna_grupo],alpha=self.alpha)
        tukey_df = pd.DataFrame(data=tukey_results._results_table.data[1:], columns=tukey_results._results_table.data[0])
        display(tukey_df.query("reject == True"))


    def test_dunn(self, columna_grupo):
        print(f"Aplicando el test de Dunn, post-hoc no paramétrico\n".upper()) 
        resultado_dunn = sp.posthoc_dunn(self.df, val_col=self.columna_metrica, group_col=columna_grupo, p_adjust='bonferroni')

        dunn_resultados_long = resultado_dunn.stack().reset_index()

        dunn_resultados_long.columns = ['Grupo_1', 'Grupo_2', 'p-valor']

        # Convert to strings for comparison
        dunn_resultados_long['Grupo_1'] = dunn_resultados_long['Grupo_1'].astype(str)
        dunn_resultados_long['Grupo_2'] = dunn_resultados_long['Grupo_2'].astype(str)
        self.df[columna_grupo] = self.df[columna_grupo].astype(str)

        # display(dunn_resultados_long[['Grupo_1']])
        dunn_resultados_long = dunn_resultados_long[dunn_resultados_long['Grupo_1'] < dunn_resultados_long['Grupo_2']]
        
        # if self.verbose:
        print("Test de Dunn (comparaciones por pares):")
        print("Hipótesis Nula (H0): Las medianas de los grupos comparados son iguales.")
        print("Hipótesis Alternativa (H1): Las medianas de los grupos comparados son significativamente diferentes.")

        resultados = []

        # Calcula las medianas por grupo
        medianas = {grupo: self.df.loc[self.df[columna_grupo]==grupo,self.columna_metrica].median() for grupo in self.df[columna_grupo].unique()}

        # Itera sobre los resultados del test de Dunn y construye el informe
        for index, row in dunn_resultados_long.iterrows():
            meddiff = round(medianas[row['Grupo_2']] - medianas[row['Grupo_1']], 3)
            
            reject = "True" if row['p-valor'] < self.alpha else "False"

            
            # Añade los resultados en formato similar a Tukey
            resultados.append([
                row['Grupo_1'], 
                row['Grupo_2'], 
                meddiff, 
                round(row['p-valor'], 3), 
                reject
            ])

        tabla_resultados = pd.DataFrame(resultados, columns=['group1', 'group2', 'meddiff', 'p-adj', 'reject_H0'])

        print("\nLas combinaciones con diferencia significativas en sus medianas, según el test de Dunn son:")
        display(tabla_resultados.query("reject_H0 == 'True'"))
        # Genera el informe
        # print("     Multiple Comparison of Medians - Dunn Test, FWER=0.05          ")
        # print("===============================================================")
        # print("     group1            group2       meddiff p-adj   reject_H0")
        # print("---------------------------------------------------------------")

        # for i, row in tabla_resultados.iterrows():
        #     # Los caracteres de > seguidos de digitos asignan alineacion derecha y padding de espacios al string
        #     print(f"{row['group1']:>14} {row['group2']:>20} {row['meddiff']:>5} {row['p-adj']:>7} {row['reject_H0']:>9}")
        #     print("---------------------------------------------------------------")
