export type FieldSize = 'sm' | 'md' | 'lg';

export interface FieldProps extends Omit<React.InputHTMLAttributes<HTMLInputElement>, 'size'> {
  /** Size preset controlling height and font size. */
  size?: FieldSize;
  /** Leading icon element. */
  icon?: React.ReactNode;
  /** Trailing element (e.g. a button or badge). */
  rightElement?: React.ReactNode;
}
