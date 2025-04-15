from base64 import b64decode
import json

from google.adk.agents import Agent
import requests

import global_var

REGISTRIES = {
    "Florianópolis": {
        "id": "A",
        "name": "1º Tabelionato de Notas e 3º Ofício de Protestos",
        "address": "Rua Emílio Blum, 131 - Edifício Hantei Office Building - Centro - 88020-010, Florianopolis - SC",
        "open_at": "09:00",
        "close_at": "18:00",
        "phone": "(48)3224-2407",
    },
    "São Paulo": {
        "id": "B",
        "name": "Cartório Paulista - 2º Cartório de Notas de São Paulo",
        "address": " Av. Paulista, 1776 - Bela Vista, São Paulo - SP, 01310-200",
        "open_at": "09:00",
        "close_at": "17:00",
        "phone": "(11)3357-8844",
    },
}

AVAILABLE_DATES = {
    "Florianópolis": [
        {
            "day": "16",
            "month": "04",
            "available_times": ["09:00", "10:00", "11:00", "17:00"],
        },
        {
            "day": "17",
            "month": "04",
            "available_times": [
                "09:00",
                "10:00",
                "11:00",
                "12:00",
                "13:00",
                "14:00",
                "15:00",
                "16:00",
                "17:00",
            ],
        },
        {
            "day": "18",
            "month": "04",
            "available_times": [
                "09:00",
                "13:00",
                "14:00",
                "15:00",
                "16:00",
                "17:00",
            ],
        },
        {
            "day": "19",
            "month": "04",
            "available_times": [
                "09:00",
                "13:00",
                "14:00",
                "15:00",
                "17:00",
            ],
        },
    ],
    "São Paulo": [
        {
            "day": "16",
            "month": "04",
            "available_times": ["09:00", "10:00", "11:00", "17:00"],
        },
        {
            "day": "17",
            "month": "04",
            "available_times": [
                "09:00",
                "10:00",
                "11:00",
                "12:00",
                "13:00",
                "14:00",
                "15:00",
                "16:00",
                "17:00",
            ],
        },
        {
            "day": "18",
            "month": "04",
            "available_times": [
                "09:00",
                "13:00",
                "14:00",
                "15:00",
                "16:00",
                "17:00",
            ],
        },
        {
            "day": "19",
            "month": "04",
            "available_times": [
                "09:00",
                "13:00",
                "14:00",
                "15:00",
                "17:00",
            ],
        },
    ],
}


def obtem_cartorio(cidade: str) -> dict:
    """
    Retorna um cartório presente na cidade especificada.

    Args:
        cidade (str): O nome da cidade para a qual queremos obter o cartório
    Returns:
        dict[str, any]: um dicionário python no seguinte formato:

        {
            "id": "X", # Identificador único do cartório
            "name": "Nome do cartório",
            "address": "Endereço do Cartório",
            "open_at": "HH:MM", # Hora em que o cartório abre
            "close_at": "HH:MM", # Hora em que o cartório fecha
            "phone": "(XX) XXXX-XXXX", # Telefone do cartório.
        }

        Caso o cartório não seja encontrado, é retornado apenas {"status": "Não encontrado"}
    """
    print(f"obtem_cartorio: {cidade=}")

    return str(REGISTRIES.get(cidade, {"status": "Não encontrado"}))


def obtem_cartorios_disponiveis() -> dict:
    """
    Retorna os cartórios dispońiveis.

    Returns:
        dict[str, any]: um dicionário python no seguinte formato:

        {
            "cidade A": {
                "id": "X", # Identificador único do cartório
                "name": "Nome do cartório",
                "address": "Endereço do Cartório",
                "open_at": "HH:MM", # Hora em que o cartório abre
                "close_at": "HH:MM", # Hora em que o cartório fecha
                "phone": "(XX) XXXX-XXXX", # Telefone do cartório.
            },
            "cidade B": {
                "id": "X", # Identificador único do cartório
                "name": "Nome do cartório",
                "address": "Endereço do Cartório",
                "open_at": "HH:MM", # Hora em que o cartório abre
                "close_at": "HH:MM", # Hora em que o cartório fecha
                "phone": "(XX) XXXX-XXXX", # Telefone do cartório.
            }
        }


        Caso o cartório não seja encontrado, é retornado apenas {"status": "Não encontrado"}
    """
    return str(REGISTRIES)


def obtem_horarios_disponiveis(cidade: str) -> list[dict]:
    """
    Obtém os horários dispońiveis de um cartório presente na cidade desejada.

    Args:
        cidade (str): nome da cidade onde o cartório está.
    Returns:
        Uma lista de dias, contendo os horários dispońiveis do cartório.

        Exemplo:
        [
            {
                "day": "16", # Dia do horário
                "month": "04", # Mês do horário

                # Tempos dispońiveis para agendamento.
                "available_times": ["09:00", "10:00", "11:00", "17:00"],
            },
            {
                "day": "17",
                "month": "04",
                "available_times": [
                    "09:00",
                    "10:00",
                    "11:00",
                    "12:00",
                    "13:00",
                    "14:00",
                    "15:00",
                    "16:00",
                    "17:00",
                ],
            },
        ]

        Caso o cartório não existe, é retornada uma lista vazia.
    """
    print(f"obtem_horarios_disponiveis: {cidade=}")
    return str(AVAILABLE_DATES.get(cidade, []))


def is_available(cidade: str, dia: str, mes: str, horario: str) -> bool | None:
    """
    Verifica se o cartório presente na cidade especificada está dispońivel para atendimento
    no momento especificado.

    Args:
        cidade (str): cidade na qual o cartório está
        dia: dia do atendimento
        mes: mês do atendimento
        horario: hora do atendimento
    Returns:
        bool indicando se o horário especificado está livre para atendimento ou não.

        Caso o cartório não exista, retorna None.
    """
    print(f"is_available: {cidade=}, {dia=}, {mes=}, {horario=}")

    if cidade not in AVAILABLE_DATES:
        return "None"

    for t in AVAILABLE_DATES[cidade]:
        if t["day"] == dia.strip() and t["month"] == mes.strip():
            return str(horario.strip() in t["available_times"])
    return "False"


def realizar_agendamento(cidade: str, dia: str, mes: str, horario: str) -> str:
    """
    Realiza o agendamento do atendimento no cartório específico.

    Args:
        cidade (str): Cidade onde o cartório está.
        dia (str): dia do agendamento
        mes (str): mês do agendamento
        horario: horário do agendamento
    Returns:
        {
            "success": bool, # flag indicando se o agendamento foi um sucesso.
            "error_message": "err" #mensagem de erro indicando a causa do erro.
        }
    """
    if is_available(cidade, dia, mes, horario) != "True":
        return str({"success": False, "error_message": "Não disponível"})

    token = global_var.var.get()

    id_token: str = token["id_token"]
    id_token_payload = id_token.split(".")[1]

    payload = json.loads(b64decode(id_token_payload + "=="))
    email = payload["email"]

    access_token = token["access_token"]

    headers = {
        "Authorization": f"Bearer {access_token}",
        "Content-Type": "application/json",
    }

    horario_end = f"{int(horario.split(':')[0]) + 1}:{horario.split(':')[1]}"
    event = {
        "summary": f"Atendimento em cartório na cidade {cidade}",
        "location": cidade,
        "description": f"Atendimento em cartório",
        "start": {
            "dateTime": f"2025-{mes}-{dia}T{horario}:00-03:00",
            "timeZone": "America/Sao_Paulo",
        },
        "end": {
            "dateTime": f"2025-{mes}-{dia}T{horario_end}:00-03:00",
            "timeZone": "America/Sao_Paulo",
        },
        "attendees": [
            {"email": email},
        ],
        "reminders": {
            "useDefault": False,
            "overrides": [
                {"method": "email", "minutes": 60},
                {"method": "popup", "minutes": 10},
            ],
        },
    }

    response = requests.post(
        "https://www.googleapis.com/calendar/v3/calendars/primary/events",
        headers=headers,
        json=event,
    )

    if response.ok:
        return str({"success": True, "link": response.json().get("htmlLink")})
    else:
        return str({"success": False, "error_message": response.text})


root_agent = Agent(
    name="weather_time_agent",
    model="gemini-2.0-flash",
    description=("Agente responsável por agendar atendimentos em cartórios"),
    instruction=(
        "Você é um agente capaz de agendar atendimentos em cartórios presentes em várias cidades do Brasil."
        " Utilize uma linguagem gentil e solícita, sempre com o intuito de ajudar os usuários."
        " Apenas responda perguntas relacionadas a cartórios. Caso o usuário tente falar de outro assunto, respeitosamene guie a conversa de volta para cartórios."
        " Para agendar um atendimento, siga o seguinte fluxo:\n"
        " 1. Pergunte ao usuário qual a cidade na qual ele deseja marcar atendimento em um cartório.\n"
        " 2. Verifica se há cartório em tal cidade.\n"
        " 3. Se não houver cartório nessa cidade, peça por outra cidade ao usuário. Se houver cartório, pergunte em que momento o usuário deseja realizar o agendamento. Garante que tal horário está dentro do intervalo no qual o cartório está aberto.\n"
        " 4. Se o cartório estiver dispońivel nesse horário, realize o agendamento. Senão, peça por outro horário."
    ),
    tools=[
        obtem_cartorio,
        obtem_cartorios_disponiveis,
        obtem_horarios_disponiveis,
        is_available,
        realizar_agendamento,
    ],
)
