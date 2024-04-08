import tkinter as tk
from tkinter import filedialog, Text, Scrollbar, Listbox, messagebox
import os
from tkinter.simpledialog import askstring

# Variable global para almacenar el directorio actual seleccionado
directorio_codigos = ""

# Función para seleccionar una carpeta base
def seleccionar_carpeta():
    global directorio_codigos
    directorio_codigos = filedialog.askdirectory()
    cargar_archivos()

# Función recursiva para cargar archivos Python de una carpeta y subcarpetas
def cargar_archivos():
    lista_archivos.delete(0, tk.END)  # Limpiar lista actual
    for root, dirs, files in os.walk(directorio_codigos):
        for file in files:
            if file.endswith('.py'):
                ruta_relativa = os.path.relpath(os.path.join(root, file), directorio_codigos)
                lista_archivos.insert(tk.END, ruta_relativa)

# Mostrar contenido de archivo seleccionado y permitir copia
def mostrar_contenido(event):
    if not lista_archivos.curselection():
        return
    archivo_seleccionado = lista_archivos.get(lista_archivos.curselection())
    ruta_completa = os.path.join(directorio_codigos, archivo_seleccionado)
    with open(ruta_completa, "r", encoding="utf-8") as file:
        contenido = file.read()
        contenido_archivo.delete(1.0, tk.END)
        contenido_archivo.insert(tk.END, contenido)

# Función para copiar contenido al portapapeles
def copiar_portapapeles():
    root.clipboard_clear()
    root.clipboard_append(contenido_archivo.get("1.0", tk.END))
    messagebox.showinfo("Copiado", "El contenido ha sido copiado al portapapeles.")

root = tk.Tk()
root.title("Buscador de Códigos Python")

# Botón para seleccionar carpeta
boton_seleccionar_carpeta = tk.Button(root, text="Seleccionar Carpeta", command=seleccionar_carpeta)
boton_seleccionar_carpeta.pack()

lista_archivos = Listbox(root, width=100, height=20)
lista_archivos.pack()
lista_archivos.bind('<<ListboxSelect>>', mostrar_contenido)

scrollbar = Scrollbar(root, orient="vertical")
scrollbar.config(command=lista_archivos.yview)
scrollbar.pack(side="right", fill="y")

lista_archivos.config(yscrollcommand=scrollbar.set)

contenido_archivo = Text(root, width=100, height=20)
contenido_archivo.pack()

# Botón para copiar al portapapeles
boton_copiar = tk.Button(root, text="Copiar al Portapapeles", command=copiar_portapapeles)
boton_copiar.pack()

root.mainloop()
