/**
 * Validace názvů souborů a složek napříč Windows, Linux a macOS.
 * Použit průnik pravidel: povolený je jen znak, který je povolen na všech třech OS.
 *
 * Zakázané znaky (alespoň na jednom OS):
 * - Windows: \ / : * ? " < > |
 * - Linux:   /
 * - macOS:   / :
 * + řídicí znaky (0x00-0x1F, 0x7F)
 * + název nesmí končit mezerou ani tečkou (Windows)
 */

/** Znaky zakázané v názvu (souboru i složky) na všech třech OS */
const DISALLOWED_CHARS = /[\\/:*?"<>|\x00-\x1F\x7F]/;

/**
 * Regulární výraz pro jeden segment cesty (název souboru nebo složky).
 * - Povoleny pouze znaky povolené na Windows, Linux i macOS.
 * - Název nesmí končit mezerou ani tečkou.
 * - Název musí mít alespoň jeden znak.
 */
export const FILE_OR_FOLDER_NAME_REGEX = /^[^\\/:*?"<>|\x00-\x1F\x7F]*[^ \\.\\/:*?"<>|\x00-\x1F\x7F]$/;

/**
 * Ověří, zda je název platný na Windows, Linux i macOS.
 * @param {string} name - Název souboru nebo složky (jeden segment, bez cest)
 * @returns {boolean}
 */
export function isValidFileOrFolderName(name) {
  if (typeof name !== 'string' || name.length === 0) return false;
  return FILE_OR_FOLDER_NAME_REGEX.test(name);
}

/**
 * Na Windows jsou rezervované názvy: CON, PRN, AUX, NUL, COM1–COM9, LPT1–LPT9 (bez ohledu na velikost).
 * Tato funkce kontroluje jen znaky; rezervované názvy je vhodné kontrolovat zvlášť při zápisu na Windows.
 */
export const WINDOWS_RESERVED_NAMES = /^(?:CON|PRN|AUX|NUL|COM[1-9]|LPT[1-9])$/i;

export default FILE_OR_FOLDER_NAME_REGEX;
